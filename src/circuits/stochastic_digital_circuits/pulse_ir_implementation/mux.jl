struct PulseBitMux <: ConcretePulseIRPrimitive
    gate::PulseIR.ConcreteAsyncOnGate
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

PulseIR.output_windows(pbm::PulseBitMux, d::Dict{Input, Window}) =
    PulseIR.output_windows(implement_twice(pbm), d)
PulseIR.valid_strict_inwindows(pbm::PulseBitMux, d::Dict{Input, Window}) =
    PulseIR.valid_strict_inwindows(implement_twice(pbm), d)

struct PulseMux <: ConcretePulseIRPrimitive
    mux::Mux
    gate::PulseIR.ConcreteAsyncOnGate
end
Circuits.abstract(m::PulseMux) = m.mux
Circuits.target(::PulseMux) = Spiking()
Circuits.inputs(m::PulseMux) = implement_deep(inputs(implement(m, Spiking())), Spiking())
Circuits.outputs(m::PulseMux) = implement_deep(outputs(implement(m, Spiking())), Spiking())
Circuits.implement(m::PulseMux, ::Spiking) = OneHotMux(m.mux, PulseBitMux(m.gate))

PulseIR.output_windows(m::PulseMux, d::Dict{Input, Window}) =
    PulseIR.output_windows(implement_twice(m), d)
PulseIR.valid_strict_inwindows(m::PulseMux, d::Dict{Input, Window}) =
    PulseIR.valid_strict_inwindows(implement_twice(m), d)