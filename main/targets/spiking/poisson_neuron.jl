import .SSim: NextSpikeTrajectory, OnOffState, EmptyState, EmptyTrajectory
using Distributions: Exponential
exponential(rate) = rand(Exponential(1/rate))

struct PoissonNeuron <: PrimitiveComponent{Spiking}
    rate::Float64
end
inputs(::PoissonNeuron) = CompositeValue((on=SpikeWire(), off=SpikeWire()))
outputs(::PoissonNeuron) = CompositeValue((out=SpikeWire(),))

SSim.initial_state(::PoissonNeuron) = OnOffState(false)

SSim.next_spike(::PoissonNeuron, t::NextSpikeTrajectory) = :out

SSim.extend_trajectory(n::PoissonNeuron, s::OnOffState, ::SSim.EmptyTrajectory) = NextSpikeTrajectory(s.on ? exponential(n.rate) : Inf)
SSim.extend_trajectory(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory) = t

SSim.advance_time_by(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory, ΔT) =
    let remaining_time = t.time_to_next_spike - ΔT
        if remaining_time == 0
            (s, EmptyTrajectory(), (:out,))
        elseif remaining_time > 0
            (s, NextSpikeTrajectory(remaining_time), ())
        else
            SSim.advancing_too_far_error(t, ΔT)
        end
    end

SSim.receive_input_spike(p::PoissonNeuron, s::OnOffState, ::SSim.Trajectory, inputname) =
    if inputname === :on
        (OnOffState(true), EmptyTrajectory(), ())
    elseif inputname === :off
        (OnOffState(false), EmptyTrajectory(), ())
    else
        error("Unrecognized input name: $inputname")
    end