module SpikingSimulator
using Gen: exponential
# TODO: move the details of `PoissonNeuron` out of the simulator and into `targets/spiking/spiking.jl`
import ..PrimitiveComponent, ..CompositeComponent, ..Spiking, ..PoissonNeuron
const PrimComp = PrimitiveComponent{>:Spiking}

#=
I'm starting to think maybe the right way to implement this is to have a `Trajectory` type
instead of or in addition to the `State` type.  At any time, a `Component` has a `Trajectory`
which specifies a trajectory of state changes and spiking for that component over some period of time,
assuming that no input spikes are received.

Instead of the current `next_spike` function, there will be a function called `extend_trajectory_to_next_spike`.
(Or `extend_trajectory_to_next_spike(...; max_time_to_extend)` where we stop extending the trajectory
after `max_time_to_extend`, even if no spike has been encountered.  This will be important since we
may have a component with lots of internal recurrent activity which only outputs rarely, where we won't
want to simulate the internal activity for long before checking if other components are emitting spikes.)

`extend_trajectory_to_next_spike` will still return the time until the next spike, so for any top-level component,
we can use it to see which subcomponent spikes first.  We then call `advance_time_by!(...)` to advance the "clock"
of each component to the time at which this next spike arrives.

(For a component whose stored trajectory is longer than `ΔT`, `advance_time_by!(..., ΔT)` causes the first
`ΔT` of the trajectory for a component to be deleted and advances the state the the point immediately after this.
I think it will never be the case that the stored trajectory is shorter than `ΔT`, since we always advanced
each trajectory to at least the next spike.

We probably want the signature of `advance_time_by!` to be such that it can output information about what
neurons have spiked when we advance to the time of a spike.  (Advancing to the time we had decided a spike would
occur is how we indicate that the spike is indeed emitted.)

After we do this `advance_time_by!` call, we send the spike which is outputted at this time to the places where it
is received.  (Via a `receive_input_spike!` call.)  This causes any future trajectory to be discarded (the code will
eventually recreate a new future trajectory with the knowledge that this spike came in), as well as updating
the state to reflect the incoming spike.

At the top level, to simulate a compontent, we can just keep alternating `extend_trajectory_to_next_spike!`
and `advance_time_by!` calls until we have advanced the time by the total amount of time we want to simulate for.


PoissonNeuron trajectory just says when the next spike is; no state.
IF trajectory also just says when the next spike is; state is the amount integrated.
LIF trajectory says when the next spike is, plus possibly has also memoized some of the deterministic
evolution of the amount integrated (due to leak).  State is the amount integrated.
=#

abstract type State end
abstract type Trajectory end
struct EmptyState <: State end
struct OnOffState <: State
    on::Bool
end
struct EmptyTrajectory <: Trajectory end
struct NextSpikeTrajectory <: Trajectory
    time_to_next_spike::Float64
end
next_spike(::Component, ::Trajectory) = error("Not implemented.")
next_spike(::Component, ::EmptyTrajectory) = nothing # no next spike is within this trajectory
next_spike(::PoissonNeuron, t::NextSpikeTrajectory) = (:out, t.time_to_next_spike)

# mutable struct Simulation
#     comp::Component
#     state::State
#     trajectory::Trajectory
# end

# Simuation(comp) = Simulation(comp, initial_state(comp), EmptyTrajectory())




extend_trajectory_to_next_spike(::Component, ::State, ::Trajectory; time_cap=Inf) = error("Not implemented.")
advance_time_by(::Component, ::State, ::Trajectory, time) = error("Not implemented.")
receive_input_spike(::Component, ::State, ::Trajectory) = error("Not implemented.")

initial_state(::PoissonNeuron) = OnOffState(false)
extend_trajectory_to_next_spike(n::PoissonNeuron, s::OnOffState, ::EmptyTrajectory; time_cap) =
    (s, NextSpikeTrajectory(s.on ? exponential(n.rate) : Inf))
extend_trajectory_to_next_spike(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory; time_cap) = (s, t)

# we should not ever need to advance time for an `EmptyTrajectory`
function advance_time_by(::PoissonNeuron, s::EmptyState, t::NextSpikeTrajectory, ΔT)
    @assert t.time_to_next_spike >= ΔT
    (s, NextSpikeTrajectory(t.time_to_next_spike - ΔT))
end

# TODO: should we not clear the trajectory?  if we always clear the trajectory, should we not have this be part of the function signature?
receive_input_spike(p::PoissonNeuron, s::OnOffState, ::Trajectory, inputname) =
    if inputname === :on
        (OnOffState(true), EmptyTrajectory())
    elseif inputname === :false
        (OnOffState(false), EmptyTrajectory())
    end

# TODO: it's probably more efficient to have each substate and subtrajectory stored next to each other
# in memory; maybe have a `Trajectory` just include the state?
struct CompositeState{S} <: State
    substates::S
end
struct CompositeTrajectory{T} <: Trajectory
    subtrajectories::T
end
function extend_trajectory_to_next_spike(c::Component, s::CompositeState, t::CompositeTrajectory; time_cap=Inf)
    time_passed = 0
    while time_passed < time_cap
        extended = map(extend_trajectory_to_next_spike!, zip(c.subcomponents, s.substates, t.subtrajectories))
        
        next_spikes = map((_c, (_s, _t)) -> next_spike(_c, _t), zip(c, extended))
        next_spiker = argmin(map((name, time) -> time, next_spikes))
        output_name, time_of_next_spike = next_spikes[next_spiker]

        time_passed += time_of_next_spike

        s = CompositeState(map((_s, _t) -> _s, extended))
        t = CompositeTrajectory(map((_s, _t) -> _t, extended))

        if emits_output(c, next_spiker, output_name)
            break;
        end






        advanced = map(zip(pairs(c.subcomponents), s.substates, t.subtrajectories)) do ((subname, subc), subs, subt)
            (subs, subt) = advance_time_by(subc, subs, subt, time_of_next_spike)

            for input_name in subcomp_receivers(c, subname, next_spiker, output_name)
                (subs, subt) = receive_input_spike(subc, subs, subt, input_name)
            end
            
            (subs, subt)
        end

    end
    return (s, t)
end
function advance_time_by()

# old thought I think doesn't apply!: 
# If the stored trajectory is less than `ΔT`, it will cause the trajectory to be deleted and the state to advance
# to the end of the trajectory, and then simulate the remainder of the `ΔT`.  (Note that the only case in
# which it )


# initial_state(::PrimComp) = error("Not implemented.")
# next_spike(::PrimComp, ::State) = error("Not implemented.")
# receive_input_spike!(::PrimComp, ::State, input_name, time) = error("Not implemented.")

# struct CompositeState{S} <: State
#     substates::S
#     # CompositeState(s::S) where {S <: Union{ # TODO: restrictions
#     #     Tuple{Vararg{<:State}},
#     #     NamedTuple{<:Any, }
#     # }}
# end
# initial_state(c::CompositeComponent) = CompositeState(map(initial_state, c.subcomponents))
# function next_spike(c::CompositeComponent)
#     childvals = map(next_spike, c.subcomponents)
#     first_to_spike_name = argmin(map((_, time) -> time, childvals))
#     _, time_to_spike = childvals[first_to_spike_name]
#     return (first_to_spike_name, time_to_spike)
# end





# function process_next_spike!(::PrimComp, ::State, time)
#     (_, time_to_spike) = next_spike(c, st)

#     return (_, time + time_to_spike)
# end

# end # module