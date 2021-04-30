"""
    OffGate

Repeats spikes in the `:in` wire unless a spike has been received in the `:off` wire recently.
"""
struct OffGate <: GenericComponent end

"""
    OnGate

Repeats spikes in the `:in` wire if a spike has been received in the `:on` wire recently.
"""
struct OnGate <: GenericComponent end

"""
    AsyncOnGate

In a memory window, outputs the same number of spikes as have been received in the `:in` wire so long
as an `:on` spike has been received within memory as well.
""" # TODO: clarify this docstring
struct AsyncOnGate <: GenericComponent end
const GatedRepeater = Union{OffGate, OnGate, AsyncOnGate}

target(::GatedRepeater) = Spiking()

Circuits.inputs(::OffGate) = NamedValues(:in => SpikeWire(), :off => :SpikeWire())
Circuits.inputs(::Union{OnGate, AsyncOnGate}) = NamedValues(:in => SpikeWire(), :on => SpikeWire())
Circuits.outputs(::GatedRepeater) = NamedValues(:out => SpikeWire())