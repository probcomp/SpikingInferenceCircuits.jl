struct PoissonOffGate <: GenericComponent
    ΔT::Float64 # time it remembers an off input
    # `R` controls the rate during ON and OFF modes.
    # "off" rate will always be ≤ exp(-R/2); "on" rate will always be ≥ exp(R/2)
    R::Float64
    M::Float64 # number of spikes which would have to be input to override an "OFF" signal
    p_F::Float64 # upper bound on probability of failing to satisfy interface
    function PoissonOffGate(args...)
        gate = PoissonOffGate(args...)
        # If this these parameters enable the given p_F, there should be a nonnegative maximum_delay
        @assert maximum_delay(gate) ≥ 0 "Invalid parameter setting"
        return gate
    end
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
            Input(:on) => CompIn(:neuron, 1),
            Input(:off) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        )
    )
#=
With these parameters, there are 2 ways we could fail:
1. A spike is emitted while it's supposed to be off
2. The delay before a spike is emitted leads to it occurring after the output window has ended

P[spikes while off within ΔT] ≤ P[Poisson(ΔT × exp(-R/2)) > 1] = 1 - exp(ΔT × exp(-R/2))
so ΔT × exp(-R/2) ≤ log(1 - P[spikes while off within ΔT])
so R ≥ 2 × log(1/ΔT × log(1 - P[spikes while off within ΔT]))
# (this explains the condition)

P[time to spike while on > T] = P[Poisson(T × exp(R/2)) = 0] = exp(T × exp(R/2))

P[fail] = 1 - (1 - P[spikes while off within ΔT])(1 - P[time to spike while on > T])
= 1 - (1 - (1 - exp(ΔT × exp(-R/2))))(1 - exp(T × exp(R/2)))
= 1 - exp(ΔT × exp(-R/2))(1 - exp(T × exp(R/2)))

so we need
p_F ≥ 1 - exp(ΔT × exp(-R/2))(1 - exp(T × exp(R/2)))
solve -->
T ≤ exp(-R/2) × log(1 + (p_F - 1) × exp(-ΔT × exp(-R/2)))
=#
maximum_delay(g::PoissonOffGate) = exp(-g.R/2) × log(1 + (g.p_F - 1) × exp(-g.ΔT × exp(-g.R/2)))

can_support_inwindows(g::PoissonOffGate, d::Dict{Input, Window}) =
    (
        # OFF window ends before IN window
        d[Input(:in)].interval.min ≥ d[Input(:off)].interval.max &&
        # Total memory of IN and OFF ≤ Neuron memory
        d[Input(:in)].interval.max - d[Input(:off)].interval.min ≤ g.ΔT &&
        # OFF has a hold window through the full output window
        d[Input(:in)].interval.max + maximum_delay(g) ≤ end_of_post_hold(d[Input(:off)]) &&
        # IN has a hold window covering the `maximum_delay` before the output begins
        d[Input(:in)].pre_hold ≥ maximum_delay(g)
    )
output_windows(g::PoissonOffGate, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Input(:in)].interval.max + maximum_delay(g)
        ),
        0., 0. # TODO: if we have extra hold time in the inputs, we can probably do better
    ))
failure_probability(g::PoissonOffGate) = g.p_F

# The input is valid so the gate was not turned off, then fed enough spikes
# to override the off signal.
is_valid_input(g::PoissonOffGate, d::Dict{Input, UInt}) =
    d[Input(:off)] == 0 || d[Input(:in)] < g.M