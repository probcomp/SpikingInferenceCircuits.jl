import .Sim: NextSpikeTrajectory, OnOffState, EmptyState, EmptyTrajectory
import Circuits: CompositeValue, PrimitiveComponent
using Distributions: Exponential
exponential(rate) = rand(Exponential(1/rate))

struct PoissonNeuron <: PrimitiveComponent{Spiking}
    rate::Float64
end
Circuits.inputs(::PoissonNeuron) = CompositeValue((on=SpikeWire(), off=SpikeWire()))
Circuits.outputs(::PoissonNeuron) = CompositeValue((out=SpikeWire(),))

Sim.initial_state(::PoissonNeuron) = OnOffState(false)

Sim.next_spike(::PoissonNeuron, t::NextSpikeTrajectory) = :out

Sim.extend_trajectory(n::PoissonNeuron, s::OnOffState, ::Sim.EmptyTrajectory) = NextSpikeTrajectory(s.on ? exponential(n.rate) : Inf)
Sim.extend_trajectory(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory) = t

Sim.advance_time_by(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory, ΔT) =
    let remaining_time = t.time_to_next_spike - ΔT
        if remaining_time == 0
            (s, EmptyTrajectory(), (:out,))
        elseif remaining_time > 0
            (s, NextSpikeTrajectory(remaining_time), ())
        else
            Sim.advancing_too_far_error(t, ΔT)
        end
    end

Sim.receive_input_spike(p::PoissonNeuron, s::OnOffState, ::Sim.Trajectory, inputname) =
    if inputname === :on
        (OnOffState(true), EmptyTrajectory(), ())
    elseif inputname === :off
        (OnOffState(false), EmptyTrajectory(), ())
    else
        error("Unrecognized input name: $inputname")
    end