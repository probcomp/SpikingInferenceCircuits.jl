### PoissonTimer
# I'm not going to make this a Pulse IR Primitive
# since it's used to operate on amounts of time, not on
# spike counts.  (That said, couldn't we say the same about WTA?)

# Anyway, so long as this is only used in Pulse IR primitives,
# this seems reasonable.

struct PoissonTimer <: GenericComponent
    ΔT::Float64 # amount of time to time in expectation
    n_spikes::Int # more spikes → more precise estimate of `ΔT` is timed
    ti_params::Tuple{Float64, Float64, Float64, Float64}
    offrate::Float64 # higher → more likely to fail!
    memory::Float64
end
Circuits.inputs(::PoissonTimer) = NamedValues(:start => SpikeWire())
Circuits.outputs(::PoissonTimer) = NamedValues(:out => SpikeWire())
Circuits.target(::PoissonTimer) = Spiking()
Circuits.implement(t::PoissonTimer, ::Spiking) =
    CompositeComponent(
        inputs(t), outputs(t),
        (
            neuron=PoissonNeuron(
                [
                    let multiplier = (t.n_spikes/t.ΔT - t.offrate)
                        x -> min(1, x) × multiplier
                    end,
                    let multiplier = -(t.n_spikes/t.ΔT - t.offrate)
                        x -> min(1, x) × multiplier
                    end
                ], t.memory,
                let offrate = t.offrate
                    u -> max(0, u + offrate)
                end
            ),
            
            ti=PoissonThresholdedIndicator(
                t.n_spikes, t.memory, t.ti_params...
            )
        ),
        (
            Input(:start) => CompIn(:neuron, 1),
            CompOut(:neuron, :out) => CompIn(:ti, :in),
            CompOut(:ti, :out) => CompIn(:neuron, 2),
            CompOut(:ti, :out) => Output(:out)
        ),
        t
    )