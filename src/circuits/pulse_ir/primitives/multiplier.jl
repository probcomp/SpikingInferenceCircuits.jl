struct SpikeCountMultiplier <: GenericComponent
    n_inputs::Int
end
Circuits.target(::SpikeCountMultiplier) = Spiking()
Circuits.inputs(m::SpikeCountMultiplier) = IndexedValues(SpikeWire() for _=1:m.n_inputs)
Circuits.outputs(::SpikeCountMultiplier) = NamedValues(:out => SpikeWire())

# TODO: concrete version