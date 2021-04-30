# TODO: instead of this being a primitive, is there some generalization of SampleStream and WTA we could use?

"""
    PulseTheta

Pulse IR theta gate primitive.  Upon receiving a spike in the `:go` wire,
outputs a sample from the distribution induced by the counts in the `:probs` input lines.
"""
# Question: do we want a separate `PulseTheta` from the stochastic digital
# circuit `Theta`?
struct PulseTheta <: GenericComponent
    n_possibilities::Int
end
Circuits.target(::PulseTheta) = Spiking()
Circuits.inputs(θ::PulseTheta) = NamedValues(
    :probs => IndexedValues(SpikeWire() for _=1:θ.n_possibilities),
    :go => SpikeWire()
)
Circuits.outputs(θ::PulseTheta) = IndexedValues(SpikeWire() for _=1:θ.n_possibilities)