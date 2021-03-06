"""
    PoissonThresholdedIndicator(ti::ConcreteThresholdedIndicator, R::Float64)
    PoissonThresholdedIndicator(threshold, ΔT, max_delay, M, R)
"""
struct PoissonThresholdedIndicator <: ConcretePulseIRPrimitive
    ti::ConcreteThresholdedIndicator
    offrate::Float64
    onrate::Float64
end
PoissonThresholdedIndicator(threshold, ΔT, max_delay, M, offrate, onrate) =
    PoissonThresholdedIndicator(ConcreteThresholdedIndicator(threshold, ΔT, max_delay, M), offrate, onrate)

# Note that PoissonThresholdedIndicator with M will output again once `M` spikes have been input,
# once the initial `threshold` have been delivered.

# TODO: constructor where we give a failure probability & it figures out the `max_delay`
# we need to accomodate that
Circuits.abstract(t::PoissonThresholdedIndicator) = t.ti
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(t::PoissonThresholdedIndicator) = Circuits.$s(Circuits.abstract(t)))
end

# Required method for an implementation of ThresholdedIndicator
threshold(t::PoissonThresholdedIndicator) = threshold(t.ti)

Circuits.implement(t::PoissonThresholdedIndicator, ::Spiking) =
    CompositeComponent(
        inputs(t), outputs(t),
        (neuron=PoissonNeuron(
            [x -> x,
            let M = t.ti.M; x -> -M*x; end
            ], t.ti.ΔT,
            truncated_linear(t.offrate, t.onrate, t.ti.threshold - 1, t.ti.threshold)
        ),),
        (
            Input(:in) => CompIn(:neuron, 1),
            CompOut(:neuron, :out) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => Output(:out)
        ),
        t
    )

failure_probability_bound(g::ThresholdedIndicator) =
    1 - (1 - p_spikes_while_off_bound(g))*(1 - p_spikes_after_delay(g))

# same as for OffGate:
p_spikes_while_off_bound(g::ThresholdedIndicator) =
    1 - exp(-g.ti.ΔT × exp(-g.R/2))

p_spikes_after_delay(g::ThresholdedIndicator) =
    1 - exp(g.ti.max_delay × exp(g.R/2))
