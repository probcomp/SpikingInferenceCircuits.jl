struct PoissonThresholdedIndicator <: GenericComponent
    threshold::Int
    ΔT::Float64 # Neuron memory
    M::Float64 # Number of spikes needed to override & produce another spike after first spike emitted
    # `R` controls the rate during ON and OFF modes.
    # "off" rate will always be ≤ exp(-R/2); "on" rate will always be ≥ exp(R/2)
    R::Float64
    p_F::Float64 # upper bound on the probability of failing to satisfy the interface
end
Circuit.abstract(t::PoissonThresholdedIndicator) = ThresholdedIndicator(t.threshold)

for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(t::PoissonThresholdedIndicator) = Circuits.$s(Circuits.abstract(t)))
end

Circuits.implement(t::PoissonThresholdedIndicator) =
    CompositeComponent(
        inputs(t), outputs(t),
        (neuron=PoissonNeuron(
            [x -> x, x -> -t.M*x], t.ΔT,
            u -> exp(t.R * (u - t.threshold + 1/2))
        ),),
        (
            Input(:in) => CompIn(:neuron, 1),
            CompOut(:neuron, :out) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => Output(:out)
        )
    )

### Temporal Interface ###

# The ways this unit could fail are identical to those for the OFF gated repeater.
# So, by the math in the comments in that file:
maximum_delay(g::ThresholdedIndicator) = exp(-g.R/2) × log(1 + (g.p_F - 1) × exp(-g.ΔT × exp(-g.R/2)))

can_support_inwindows(t::ThresholdedIndicator, d::Dict{Input, Window}) =
    (
        d[Input(:in)].pre_hold ≥ t.ΔT &&
        interval_length(d[Input(:in)]) ≤ t.ΔT - max_delay(t)
    )
output_windows(t::ThresholdedIndicator, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Inpt(:in)].interval.max + maximum_delay(t)
        ),
        0., 0. # TODO: if we have extra hold time in the inputs, we can probably do better
    ))

failure_probability(t::ThresholdedIndicator) = t.p_F

# The input is valid unless after passing the threshold, it was fed enough spikes
# that it could override the indicator being OFF.
is_valid_input(g::ThresholdedIndicator, d::Dict{Input, UInt}) =
    d[Input(:in)] < g.threshold + g.M