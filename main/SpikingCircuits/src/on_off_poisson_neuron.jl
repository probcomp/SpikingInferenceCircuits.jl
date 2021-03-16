import .Sim: NextSpikeTrajectory, OnOffState, EmptyState, EmptyTrajectory
import Circuits: CompositeValue, PrimitiveComponent
using Distributions: Exponential
exponential(rate) = rand(Exponential(1/rate))

struct OnOffPoissonNeuron <: PrimitiveComponent{Spiking}
    rate::Float64
end
Circuits.inputs(::OnOffPoissonNeuron) = CompositeValue((on=SpikeWire(), off=SpikeWire()))
Circuits.outputs(::OnOffPoissonNeuron) = CompositeValue((out=SpikeWire(),))

Sim.initial_state(::OnOffPoissonNeuron) = OnOffState(false)

Sim.next_spike(::OnOffPoissonNeuron, t::NextSpikeTrajectory) = :out

Sim.extend_trajectory(n::OnOffPoissonNeuron, s::OnOffState, ::Sim.EmptyTrajectory) = NextSpikeTrajectory(s.on ? exponential(n.rate) : Inf)
Sim.extend_trajectory(::OnOffPoissonNeuron, s::OnOffState, t::NextSpikeTrajectory) = t

Sim.advance_time_by(::OnOffPoissonNeuron, s::OnOffState, t::NextSpikeTrajectory, ΔT) =
    let remaining_time = t.time_to_next_spike - ΔT
        if remaining_time == 0
            (s, EmptyTrajectory(), (:out,))
        elseif remaining_time > 0
            (s, NextSpikeTrajectory(remaining_time), ())
        else
            Sim.advancing_too_far_error(t, ΔT)
        end
    end

Sim.receive_input_spike(p::OnOffPoissonNeuron, s::OnOffState, ::Sim.Trajectory, inputname) =
    if inputname === :on
        (OnOffState(true), EmptyTrajectory(), ())
    elseif inputname === :off
        (OnOffState(false), EmptyTrajectory(), ())
    else
        error("Unrecognized input name: $inputname")
    end