import Circuits: IndexedValues

"""
    IntegratingPoisson <: PrimitiveComponent{Spiking}
    IntegratingPoisson(weights::Vector{Float64}, bias::Float64, rate_fn::Function)

A Poisson neuron whose potential at time `t` is `bias` plus
the weighted sum of all input spikes received by `t`.
At any time, fires as a Poisson Process with rate `rate_fn(potential(t))`.
"""
struct IntegratingPoisson <: PrimitiveComponent{Spiking}
    weights::Vector{Float64}
    bias::Float64
    rate_fn::Function
end
Circuits.inputs(p::IntegratingPoisson) = IndexedValues(SpikeWire() for _=1:length(p.weights))
Circuits.outputs(p::IntegratingPoisson) = CompositeValue((out=SpikeWire(),),)

struct PotentialState <: Sim.State
    potential::Float64
end
Sim.initial_state(p::IntegratingPoisson) = PotentialState(p.bias)
Sim.next_spike(::IntegratingPoisson, ::Sim.NextSpikeTrajectory) = :out

function Sim.extend_trajectory(p::IntegratingPoisson, st::PotentialState, ::Sim.EmptyTrajectory)
    rate = p.rate_fn(st.potential)
    @assert rate >= 0 "IntegratingPoisson rate_fn returned a negative value!  Rates must be nonnegative."
    return NextSpikeTrajectory(exponential(rate))
end
Sim.extend_trajectory(::IntegratingPoisson, ::PotentialState, t::Sim.NextSpikeTrajectory) = t

Sim.advance_time_by(::IntegratingPoisson, s::PotentialState, t::NextSpikeTrajectory, ΔT) =
    Sim.advance_without_statechange(s, t, ΔT)

Sim.receive_input_spike(p::IntegratingPoisson, s::PotentialState, ::Sim.Trajectory, inidx) =
    let new_potential = s.potential + p.weights[inidx]
        if p.rate_fn(new_potential) == Inf
            prev_rate_msg = p.rate_fn(s.potential) == Inf ? " (In this case, the rate was Inf before this spike was received as well.) " : ""
            @warn("IntegratingPoisson just recieved an input spike; after this, the rate is `Inf`. $prev_rate_msg This can cause an infinite number of spikes in a single instant.")
            (PotentialState(new_potential), EmptyTrajectory, (:out,))
        else
            (PotentialState(new_potential), EmptyTrajectory(), ())
        end
    end