struct PoissonOffGate <: GenericComponent
    ΔT::Float64 # time it remembers an off input
    # `R` controls the rate during ON and OFF modes.
    # "off" rate will always be ≤ exp(-R/2); "on" rate will always be ≥ exp(R/2)
    R::Float64
    M::Float64 # number of spikes which would have to be input to override an "OFF" signal
    max_delay::Float64
end
Circuits.abstract(::PoissonOffGate) = OffGate()

for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonOffGate) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(g::PoissonOffGate) =
    CompositeComponent(
        inputs(g), outputs(g),
        (neuron=PoissonNeuron([
            x -> x, x -> -g.M*x, x -> -x
        ], g.ΔT, u -> exp(g.R*(u - 1/2))),),
        (
            Input(:in) => CompIn(:neuron, 1),
            Input(:off) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        )
    )

failure_probability_bound(g::PoissonOffGate) =
    1 - (1 - p_spikes_while_off_bound(g))*(1 - p_doesnt_spike_by_delay_bound(g))

p_spikes_while_off_bound(g::PoissonOffGate) =
    1 - exp(-g.ΔT × exp(-g./R/2))

# Derived on page 76 of notebook
p_doesnt_spike_by_delay_bound(g::PosisonOffGate) =
    let α = g.max_delay × (exp(g.R) - 1)
        1 - (1 - exp(-α × exp(-g.R/2)))^(g.M - 1)
    end

can_support_inwindows(g::PoissonOffGate, d::Dict{Input, Window}) =
    (
        # OFF window ends before IN window
        d[Input(:in)].interval.min ≥ d[Input(:off)].interval.max &&
        # Total memory of IN and OFF ≤ Neuron memory
        d[Input(:in)].interval.max - d[Input(:off)].interval.min ≤ g.ΔT &&
        # OFF has a hold window through the full output window
        d[Input(:in)].interval.max + g.max_delay ≤ end_of_post_hold(d[Input(:off)]) &&
        # IN has a hold window covering the `maximum_delay` before the output begins
        d[Input(:in)].pre_hold ≥ g.max_delay
    )
output_windows(g::PoissonOffGate, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Input(:in)].interval.max + g.max_delay
        ),
        0., 0. # TODO: if we have extra hold time in the inputs, we can probably do better
    ))

# The input is valid so long as there are fewer than M input spikes
is_valid_input(g::PoissonOffGate, d::Dict{Input, UInt}) = d[Input(:in)] < g.M