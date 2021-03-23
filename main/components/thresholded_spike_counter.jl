THRESHOLDED_COUNTER_OFF_RATE = 1e-5
THRESHOLDED_COUNTER_ON_RATE = 1e5

"""
    ThresholdedCounter <: GenericComponent
    ThresholdedCounter(threshold)

Concrete component for `Spiking`; fires once it has recieved `threshold` spikes,
then resets.
"""
struct ThresholdedCounter <: GenericComponent
    threshold::Int
end
Circuits.target(::ThresholdedCounter) = Spiking()
Circuits.inputs(::ThresholdedCounter) = NamedValues(:in => SpikeWire())
Circuits.outputs(::ThresholdedCounter) = NamedValues(:out => SpikeWire())

Circuits.implement(t::ThresholdedCounter, ::Spiking) =
    CompositeComponent(
        inputs(t), outputs(t),
        (neuron=(
            let weight = log(THRESHOLDED_COUNTER_ON_RATE) - log(THRESHOLDED_COUNTER_OFF_RATE)
                IntegratingPoisson(
                    [weight, -(weight * t.threshold)],
                    log(THRESHOLDED_COUNTER_OFF_RATE) - (weight * (t.threshold - 1)),
                    exp
                )
            end
        ),),
        (
            Input(:in) => CompIn(:neuron, 1),
            CompOut(:neuron, :out) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => Output(:out)
        ),
        t
    )