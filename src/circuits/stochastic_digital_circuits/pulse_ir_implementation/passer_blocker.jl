Circuits.implement(b::BitBlockerPasser{is_blocker}, ::Spiking) where {is_blocker} =
    CompositeComponent(
        NamedValues(pb(b) => SpikeWire(), :val => SpikeWire()),
        NamedValues(:out => SpikeWire()),
        (gate=(is_blocker ? PulseIR.OffGate() : PulseIR.OnGate()),),
        (
            Input(pb(b)) => CompIn(:gate, is_blocker ? :off : :on),
            Input(:val) => CompIn(:gate, :in),
            CompOut(:gate, :out) => Output(:out)
        ), b
    )