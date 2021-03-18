const IPOISSONGATE_REPEATING_RATE = 1000.0

"""
    IPoissonGatedRepeater <: GenericComponent
    IPoissonGatedRepeater()
    
Rapidly repeats given input spikes until a spike is recieved in the `off`
wire, then stops outputting forever.
"""
struct IPoissonGatedRepeater <: GenericComponent end
Circuits.inputs(::IPoissonGatedRepeater) = NamedValues(
        :in => SpikeWire(),
        :off => SpikeWire()
    )
Circuits.outputs(r::IPoissonGatedRepeater) = NamedValues(:out => SpikeWire())
Circuits.target(::IPoissonGatedRepeater) = Spiking()
Circuits.implement(r::IPoissonGatedRepeater, ::Spiking) =
    CompositeComponent(
        inputs(r), outputs(r),
        (IntegratingPoisson([+IPOISSONGATE_REPEATING_RATE, -IPOISSONGATE_REPEATING_RATE, -Inf], 0., identity),),
        (
            Input(:in) => CompIn(1, 1),
            CompOut(1, :out) => CompIn(1, 2),
            CompOut(1, :out) => Output(:out),
            Input(:off) => CompIn(1, 3)
        )
    )