struct PoissonAsyncOnGate <: GenericComponent
    ΔT::Float64 # time it remembers an off input
    # `R` controls the rate during ON and OFF modes.
    # "off" rate will always be ≤ exp(-R/2); "on" rate will always be ≥ exp(R/2)
    R::Float64
    M::Float64 # number of spikes which would have to be input to pass through without an `on` input
    p_F::Float64 # upper bound on probability of failing to satisfy interface
    max_delay::Float64
end
Circuits.abstract(::PoissonAsyncOnGate) = AsyncOnGate()

for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonAsyncOnGate) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(g::PoissonOffGate) =
    CompositeComponent(
        inputs(g), outputs(g),
        (neuron=PoissonNeuron([
            x -> x, x -> g.M*min(x, 1), x -> -x
        ], g.ΔT, u -> exp(g.R*(u - g.M + 1/2))),),
        (
            Input(:in) => CompIn(:neuron, 1),
            Input(:on) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        )
    )

### Temporal inteface ###

# the next 4 functions are copied from `OffGate`
failure_probability_bound(g::PoissonAsyncOnGate) =
    1 - (1 - p_spikes_while_off_bound(g))*(1 - p_doesnt_spike_by_delay_bound(g))

p_spikes_while_off_bound(g::PoissonAsyncOnGate) =
    1 - exp(-g.ΔT × exp(-g./R/2))

# Derived on page 76 of notebook
p_doesnt_spike_by_delay_bound(g::PoissonAsyncOnGate) =
    let α = g.max_delay × (exp(g.R) - 1)
        1 - (1 - exp(-α × exp(-g.R/2)))^(g.M - 1)
    end

is_valid_input(g::PoissonAsyncOnGate, d::Dict{Input, UInt}) = d[Input(:in)] < g.M

can_support_inwindows(g::PoissonAsyncOnGate, d::Dict{Input, Window}) =
    (
       d[Input(:on)].pre_hold ≥ g.ΔT &&
       d[Input(:in)].pre_hold ≥ g.max_delay &&
       interval_length(d[Input(:in)]) ≤ g.ΔT - g.max_delay
    )
output_windows(g::PoissonAsyncOnGate, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Input(:in)].interval.max + g.max_delay
        ),
        0.0, 0.0 # TODO: if we have extra hold time in the inputs, we can probably do better
    ))