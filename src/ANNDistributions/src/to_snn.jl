"""
Compile ANN → sSNN.

Circuit types:
- FullyConnectedANN : Direct compilation of each neuron in an ANN to a neuron in a SNN.
Configured so that the first layer receives binary input (either 1 spike or 0 spike), and multiple input
spikes are treated as 1.  Will initially output incorrect rates until internal rates stabilize;
furthermore, will quickly forget inputted values unless the inputs are repeated.

- FullyConnectedANNWithDelay : FullyConnectedANN which enforces a delay before the output layer
begins spiking, to ensure that the rate values have stabilized before the output layer begins spiking.
"""

using Circuits, SpikingCircuits, SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

# For now, specialize this to the spiking target - we can generalize later if we want
struct FullyConnectedANN <: GenericComponent
    layers                  :: Vector{<:Flux.Dense}
    neuron_memory           :: Float64
    internal_neuron_maxrate :: Float64
    output_neuron_maxrate   :: Float64
end
n_input_lines(n      :: FullyConnectedANN) = size(n.layers[1].weight)[2]
n_output_lines(n     :: FullyConnectedANN) = size(last(n.layers).weight)[1]
Circuits.target(     :: FullyConnectedANN) = Spiking()
Circuits.inputs(n    :: FullyConnectedANN) = IndexedValues(SpikeWire() for _=1:n_input_lines(n))
Circuits.outputs(n   :: FullyConnectedANN) = IndexedValues(SpikeWire() for _=1:n_output_lines(n))
Circuits.implement(n :: FullyConnectedANN, ::Spiking) = CompositeComponent(
    inputs(n), outputs(n),
    Tuple( # layers
        IndexedComponentGroup(
            neuron(n.neuron_memory, layer.weight[i, :], layer.bias[i], layer.σ, layeridx == 1, layeridx == length(n.layers), n.internal_neuron_maxrate, n.output_neuron_maxrate)
            for i=1:size(layer.weight)[1]
        )
        for (layeridx, layer) in enumerate(n.layers)
    ),
    (
        # input to first layer:
        (
            Input(i) => CompIn(1 => j, i)
            for i=1:size(n.layers[1].weight)[2], j=1:size(n.layers[1].weight)[1]
        )...,

        # connect intermediate layers:
        Iterators.flatten(
            (
                CompOut(l - 1 => inidx, :out) => CompIn(l => outidx, inidx)
                for outidx=1:size(n.layers[l].weight)[1] for inidx=1:size(n.layers[l].weight)[2]
            )
            for l=2:length(n.layers)
        )...,

        # last layer to output
        (
            CompOut(length(n.layers) => i, :out) => Output(i)
            for i=1:size(last(n.layers).weight)[1]
        )...
    ), n
)
function neuron(ΔT, W, b, σ, is_first_layer, is_last_layer, internal_maxrate, out_maxrate)
    infn = (if is_first_layer # if first layer, gets one-hot input, not a rate, so we don't normalize by ΔT
                w -> c -> w*min(c, 1) / internal_maxrate
           else
                w -> c -> w*c/ΔT / internal_maxrate
           end)
    ratemult = is_last_layer ? out_maxrate : internal_maxrate
    return SIC.PulseIR.PoissonNeuron([infn(w) for w in W], ΔT, u -> σ(u + b) * ratemult)
end

###
struct FullyConnectedANNWithDelay <: GenericComponent
    ann               :: FullyConnectedANN
    network_memory    :: Float64
    delay             :: Float64
    timer_params      :: Tuple{Real, Tuple{Real, Real, Real, Real}, Real}
    timer_memory_mult :: Real# how many times the `delay` should the timer remember?
end
FullyConnectedANNWithDelay(ann::FullyConnectedANN, network_memory::Real, timer_params, timer_memory_mult) = FullyConnectedANNWithDelay(
    ann, network_memory, ann.neuron_memory * length(ann.layers), timer_params, timer_memory_mult
)
Circuits.target(n::FullyConnectedANNWithDelay) = target(n.ann)
Circuits.inputs(n::FullyConnectedANNWithDelay) = inputs(n.ann)
Circuits.outputs(n::FullyConnectedANNWithDelay) = outputs(n.ann)

function add_delay_line(neuron_layer::Circuits.ComponentGroup)
    return IndexedComponentGroup(
        add_delay_line(neuron)
        for neuron in neuron_layer.subcomponents
    )
end
add_delay_line(neuron::InputFunctionPoisson) = InputFunctionPoisson(
    (neuron.input_functions..., c -> 1000 * min(c, 1)),
    (neuron.memories..., neuron.memories[1]),
    u -> neuron.rate_fn(u - 1000)
)

function Circuits.implement(n::FullyConnectedANNWithDelay, ::Spiking)
    ann_impl = implement(n.ann, Spiking())
    n_inputs = length(inputs(ann_impl))
    n_inputs_to_last_layer = inputs(first(last(ann_impl.subcomponents).subcomponents)) |> length

    # Create a near-duplicate of `ann_impl`, but with an additional input line which needs to receive
    # a pulse before the circuit begins outputting values
    # Currently this last input is just indexed so we can exactly reuse the edges from `ann_impl`, but this is arguably a bit sloppy.
    ann_circuit = CompositeComponent(
        # add one more input which recieves the :go signal
        IndexedValues(SpikeWire() for _=1:(n_inputs + 1)),
        outputs(ann_impl),
        Tuple(
            i == length(ann_impl.subcomponents) ? add_delay_line(layer) : layer
            for (i, layer) in enumerate(ann_impl.subcomponents)
        ),
        (
            Circuits.get_edges(ann_impl)...,
            # Feed the delay line into each of the outputs
            (
                Input(n_inputs + 1) => CompIn(length(ann_impl.subcomponents) => i, n_inputs_to_last_layer + 1)
                for i=1:n_output_lines(n.ann)
            )...
        )
    )

    return CompositeComponent(
        inputs(n), outputs(n),
        (
            ann         = ann_circuit,
            # Input layer to repeat input spikes for `network_memory`
            # Currently rate is set so we expect to get 10 spikes per neuron memory
            input_layer = IndexedComponentGroup(
                InputFunctionPoisson((c -> min(c, 1.),), (n.network_memory,), u -> u * 10/n.ann.neuron_memory)
                for _=1:n_input_lines(n.ann)
            ),
            timer       = SIC.PulseIR.PoissonTimer(n.delay, n.timer_params..., n.timer_memory_mult * n.delay)
        ),
        (
            (
                Input(i) => CompIn(:input_layer => i, 1)
                for i=1:length(inputs(n))
            )...,
            (
                CompOut(:input_layer => i, :out) => CompIn(:ann, i)
                for i=1:length(inputs(n))
            )...,
            (
                Input(i) => CompIn(:timer, :start)
                for i=1:length(inputs(n))
            )...,
            CompOut(:timer, :out) => CompIn(:ann, length(inputs(n)) + 1),
            (
                CompOut(:ann, i) => Output(i)
                for i=1:length(outputs(n))
            )...
        )
    )
end