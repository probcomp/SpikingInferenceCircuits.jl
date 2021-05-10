# TODO: instead of this being a primitive, is there some generalization of SampleStream and WTA we could use?

"""
    PulseIR.Theta

Pulse IR theta gate primitive.  Upon receiving a spike in the `:go` wire,
outputs a sample from the distribution induced by the counts in the `:probs` input lines.
"""
struct Theta <: GenericComponent
    n_possibilities::Int
end
Circuits.target(::Theta) = Spiking()
Circuits.inputs(θ::Theta) = NamedValues(
    :probs => IndexedValues(SpikeWire() for _=1:θ.n_possibilities),
    :go => SpikeWire()
)
Circuits.outputs(θ::Theta) = IndexedValues(SpikeWire() for _=1:θ.n_possibilities)

# TODO: Concrete Theta