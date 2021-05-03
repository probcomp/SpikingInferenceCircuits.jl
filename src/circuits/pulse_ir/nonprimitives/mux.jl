# struct PulseBitMux <: GenericComponent end
# Circuits.abstract(::PulseBitMux) = BitMux()
# Circuits.target(::PulseBitMux) = Spiking()
# Circuits.inputs(::PulseBitMux) = NamedValues(
#     :sel => SpikeWire(), :value => SpikeWire()
# )
# Circuits.outputs(::PulseBitMux) = NamedValues(:out => SpikeWire())

# Circuits.implement(::PulseBitMux)

Circuits.implement(::BitMux, ::Spiking) =
    RelabeledIOComponent(AsyncOnGate(), (:in => :value, :on => :sel), (), BitMux())

# Maybe I don't want to provide this, and want the implementation of BitMux
# to go directly to a particular type of AsyncOnGate (ie. with particular parameters).