struct PulseBitMux <: ConcretePulseIRPrimitive
    gate::ConcreteAsyncOnGate
end
Circuits.abstract(::PulseBitMux) = BitMux()
Circuits.target(::PulseBitMux) = Spiking()
Circuits.inputs(::PulseBitMux) = NamedValues(
    :sel => SpikeWire(), :value => SpikeWire()
)
Circuits.outputs(::PulseBitMux) = NamedValues(:out => SpikeWire())

Circuits.implement(pbm::PulseBitMux, ::Spiking) =
    RelabeledIOComponent(pbm.gate, (:in => :value, :on => :sel), (), BitMux())

implement_twice(c) = implement(implement(c, Spiking()), Spiking())

output_windows(pbm::PulseBitMux, d::Dict{Input, Windows}) =
    output_windows(implement_twice(pbm), d)
valid_strict_inwindows(pbm::PulseBitMux, d::Dict{Input, Windows}) =
    valid_strict_inwindows(implement_twice(pbm), d)

struct PulseMux <: ConcretePulseIRPrimitive
    mux::Mux
    gate::ConcreteAsyncOnGate
end
Circuits.abstract(m::PulseMux) = m.mux
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PulseMux) = Circuits.$s(Circuits.abstract(g)))
end
Circuits.implement(m::PulseMux, ::Spiking) = OneHotMux(mux, PulseBitMux(m.gate))

output_windows(m::PulseMux, d::Dict{Input, Windows}) =
    output_windows(implement_twice(m), d)
valid_strict_inwindows(m::PulseMux, d::Dict{Input, Windows}) =
    valid_strict_inwindows(implement_twice(m), d)