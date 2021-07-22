"""
Use a compiled ANN to implement the CPT Sample / CPT Score interface.
"""

struct ANNCPTSample <: GenericComponent
    ann::FullyConnectedANNWithDelay
    input_ncategories::Tuple
end
ANNCPTSample(
    layers::Vector{<:Flux.Dense},
    neuron_memory::Real,
    network_memory::Real,
    timer_params,
    input_ncategories::Tuple
) = ANNCPTSample(FullyConnectedANNWithDelay(FullyConnectedANN(layers, neuron_memory), network_memory, timer_params), input_ncategories)
ANNCPTSample(
    chain::Flux.Chain, args...
) = ANNCPTSample(chain.layers |> collect, args...)
out_domain_size(c::ANNCPTSample) = n_output_lines(c.ann.ann)

Circuits.inputs(c::ANNCPTSample) = NamedValues(
    :in_vals => IndexedValues(
        SIC.FiniteDomainValue(n)
        for n in c.input_ncategories
    )
)
Circuits.outputs(c::ANNCPTSample) = NamedValues(
    :value => SIC.FiniteDomainValue(length(outputs(c.ann))),
    :inverse_prob => SIC.ReciprocalProbEstimate()
)

# TODO: this is sloppy!
# But for now to get the `wta` and `counter` implementations,
# we just get what the implementation is for a simple ConditionalSample
# with a simple CPT with the right output size
# This will use whatever implementaiton rule is defined for ConditionalSample
example_pulse_cond_sample(c) = implement(
    SIC.SDCs.ConditionalSample([i == 1 ? 1. : 0. for i=1:out_domain_size(c)] |> x->reshape(x, (1, :))),
    Spiking()
)
wta(c::ANNCPTSample) = SIC.SDCs.wta(
    example_pulse_cond_sample(c)
)
probcounter(c::ANNCPTSample) = SIC.SDCs.probcounter(
    example_pulse_cond_sample(c)
)

Circuits.implement(c::ANNCPTSample, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(c), Spiking()), 
        implement_deep(outputs(c), Spiking(), :value), # only implement `value`
        (
            ann = c.ann,
            # TODO; methods to get WTA and ProbCounter
            wta = wta(c),
            counter = probcounter(c)
        ),
        (
            # Inputs -> ANN
            onehots_to_ann_input(c.input_ncategories)...,

            # The remaining edges are pretty much copied exactly from `ConditionalSample`
            (
                CompOut(:ann, i) => CompIn(:wta, i)
                for i=1:out_domain_size(c)
            )...,
            Iterators.flatten(
                (
                    CompOut(:wta, i) => CompIn(:counter, :sel => i),
                    CompOut(:ann, i) => CompIn(:counter, :samples => i)
                )
                for i=1:out_domain_size(c)
            )...,
            (
                CompOut(:wta, i) => Output(:value => i)
                for i=1:out_domain_size(c)
            )...,
            CompOut(:counter, :count) => Output(:inverse_prob)
        ), c
    )
function onehots_to_ann_input(input_ncategories)
    edges = []
    for (varidx, n) in enumerate(input_ncategories)
        start = length(edges)
        for i=1:n
            val = start + i
            push!(edges,
                Input(:in_vals => varidx => i) => CompIn(:ann, val)
            )
        end
    end
    return edges
end