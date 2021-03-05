"""
    SpikingSimulator

A event-based simulator for the `Spiking` target.

At any time during simulation, each component has a `State`
and a `Trajectory`.  A `t::Trajectory` describes how the state will evolve
over the next `trajectory_length(t)` milliseconds if no input spikes are received in this time.
A trajectory may specify the next output spike which will be emmitted from this component if no inputs are received.
`next_spike(::Component, t::Trajectory)` returns `nothing` if the trajectory does not extend to a time
when the component emits an output spike, or `output_name` if it does.
If the trajectory extends to a spike emission, `trajectory_length(t)` is the time of this spike emission.

The function `extend_trajectory(::Component, ::State, t::Trajectory)` returns the a new `trajectory`
such that `trajectory_length(trajectory) > trajectory_length(t)`.  [TODO: do we want `>` or `≥`?]

The function `advance_time_by(::Component, ::State, t::Trajectory, ΔT)` returns
a tuple `(new_state, new_traj, output_spike_names)` giving a new state and trajectory
for the component obtained by advancing the time by `ΔT`, and an iterator over the names
of the outputs from this component which spiked during this time.  It is required
that `ΔT ≤ trajectory_length(t)`, so we are only advancing time to a point we have already
determined in the trajectory.  Note that this means that any spikes emitted by the component
are emitted at exactly time `ΔT`, since we require that any output spikes the trajectory `t`
includes occur at `trajectory_length(t)`.

The function `receive_input_spike(::Component, ::State, ::Trajectory, inname)`
returns a tuple `(new_state, new_traj, spiking_output_names)` of a new state and (possibly empty)
trajectory reflecting knowledge of the inputted spike.  `spiking_output_names` is an iterator
over the names of the component outputs which spike immediatly upon receiving this input spike.
This will often return an empty trajectory, since the old trajectory is rendered invalid by the knowledge
that a new spike arrived.

Currently the simulator operates on immutable objects; it may be more performant to move to mutable objects.

Note: I think that the code probably requires some condition about the order in which input spikes are received
and output spikes are emitted in one moment of time not mattering.  It may be worthwhile to articulate this at some point.
It may also be the right call to move to a simulator where nothing can happen at the same instant, and when things in theory happen
at the same instant, we instead sequence them with some tiny ϵ-milisecond delay instead.
"""
module SpikingSimulator

using Distributions: Exponential
exponential(rate) = rand(Exponential(1/rate))
using DataStructures: Queue, enqueue!, dequeue!
# TODO: move the details of `PoissonNeuron` out of the simulator and into `targets/spiking/spiking.jl`
import ..Component, ..PrimitiveComponent, ..CompositeComponent, ..Spiking, ..PoissonNeuron
import ..Output, ..Input, ..CompOut, ..CompIn, ..receivers, ..does_output
const PrimComp = PrimitiveComponent{>:Spiking}

#############
# Interface #
#############

"""
    State

A spiking component state.
"""
abstract type State end

"""
    Trajectory

A spiking component trajectory.

A `t::Trajectory` describes how the state will evolve
over the next `trajectory_length(t)` milliseconds, assuming that no input spikes are received in this time.

Invariant: if no input spikes arrive in the next `trajectory_length(t)` milliseconds, no output spikes are emitted
until `trajectory_length(t)` milliseconds have passed.  (Ie. the soonest an output spike could occur is in exactly `trajectory_length(t)` ms.)
"""
abstract type Trajectory end

"""
    initial_state(::Component)

The initial `State` for a spiking component.
"""
initial_state(::Component) = error("Not implemented.")

"""
    empty_trajectory(::Component)

A `t::Trajectory` for the spiking component such that `trajectory_length(t) == 0`.
"""
empty_trajectory(::Component) = EmptyTrajectory()

"""
    next_spike(::Component, t::Trajectory)

If the trajectory does not extend to a time when the component outputs a spike, returns `nothing`.
If the trajectory does extend to a time when the component outputs a spike, returns the name of the component
output which spikes at `trajectory_length(t)`.
"""
# TODO: do we want `next_spike`? or something like `next_spikes` in case multiple spikes happen at once?
next_spike(::Component, ::Trajectory) = error("Not implemented.")

"""
    extended_trajectory = extend_trajectory(::Component, ::State, old_trajectory::Trajectory)

Return a new trajectory for the component such that `length(extended_trajectory) > length(old_trajectory)`.

Input condition: `next_spike(old_trajectory)` must be `nothing`.  Ie. we cannot extend a trajectory
past the next output spike.
"""
extend_trajectory(::Component, ::State, ::Trajectory) = error("Not implemented.")

function advancing_too_far_error(t, ΔT)
    @assert ΔT > trajectory_length(t) "This error should only be called when `ΔT > trajectory_length(t)`!  (But `ΔT = $ΔT` and `trajectory_length(t)` = $(trajectory_length(t))"
    error("`advance_by_time` called to advance a trajectory past its length.  (Asked to extend a trajectory of length $(trajectory_length(t)) by $ΔT seconds.")
end

"""
    (new_state, new_traj, spiking_output_names) = advance_time_by(::Component, ::State, t::Trajectory, ΔT)

Returns a new state and trajectory obtained by advancing along the current trajectory by `ΔT`.  Also returns
the iterator `spiking_output_names` over the names of the component's outputs which spike within those `ΔT` seconds.

Input condition: `trajectory_length(t) ≥ ΔT`.

Note that this condition, and the invariant that a trajectory length cannot extend past the next output spike,
implies that if `spiking_output_names` is not empty, then all spikes in `spiking_output_names`
occurred at exactly `ΔT` seconds.
(Ie. if `Ts` is the time of the spike output, the trajectory length invariant tells us
`Ts ≥ trajectory_length(t)`, and thus the input condition tells us `Ts ≥ ΔT`.
Thus, if `Ts ≤ ΔT` so that `spiking_output_names` is nonempty, we must have `Ts = ΔT`.)
"""
advance_time_by(::Component, ::State, ::Trajectory, ΔT) = error("Not implemented.")

"""
    (new_state, new_traj, spiking_output_names) = advance_time_by(c::Component, ::State, ::Trajectory, ΔT, f::Function)

Same as `advance_time_by`, but calls `f(itr, ΔT)` before returning, where
`itr` is an iterator over pairs `(nested_name, outputname)`.
`nested_name` has the form `n1 => (n2 => ... => (n_n))`,
and specifies the subcomponent `c.subcomponents[n1].subcomponents[n2]....subcomponents[n_n]`.
`outputname` is an output of this nested subcomponent which spiked at `ΔT` seconds.
`nested_name` will be `nothing` to indicate a spike for this output and `n1` to indicate a spike for `c.subcomponents[n1]`.

By default, for any component other than a `CompositeComponent`, it is assumed that there are no internal spikes,
and so `f` is only called on the component output spikes.
"""
function advance_time_by(c::Component, s::State, t::Trajectory, ΔT, f::Function)
    (n, t, son) = advance_time_by(c, s, t, ΔT)
    f(((nothing, name) for name in son), ΔT)
    return (n, t, son)
end

"""
    (new_state, new_traj, spiking_output_names) = receive_input_spike(::Component, ::State, ::Trajectory, inname)

Returns a new state and trajectory consistent with the component receiving a spike in the input with name `inname`.
Also returns an iterator `spiking_output_names` over names of component outputs which spike immediately when
this input is received.

`new_traj` may be any valid trajectory for the component consistent with this spike being received.
(One common choice is to have `new_traj` simply be an empty trajectory.)
"""
receive_input_spike(::Component, ::State, ::Trajectory, inname) = error("Not implemented.")

### General State/Trajectory Types ###

"""
    EmptyState <: State

Empty state containing no information.
"""
struct EmptyState <: State end

"""
    OnOffState <: State

State denoting that a component is either on or off.
`state.on` is a boolean returning whether it is on.
"""
struct OnOffState <: State
    on::Bool
end

"""
    EmptyTrajectory <: Trajectory

A trajectory of length 0.
"""
struct EmptyTrajectory <: Trajectory end
Base.show(io::IO, t::EmptyTrajectory) = print(io, "EmptyTrajectory()")

"""
    NextSpikeTrajectory <: Trajectory

A trajectory containing the time of the next spike (given by
`traj.time_to_next_spike`).
"""
struct NextSpikeTrajectory <: Trajectory
    time_to_next_spike::Float64
end
Base.show(io::IO, t::NextSpikeTrajectory) = print(io, "NextSpikeTrajectory($(t.time_to_next_spike))")

next_spike(::Component, ::EmptyTrajectory) = nothing # no next spike is within this trajectory

trajectory_length(::EmptyTrajectory) = 0.
trajectory_length(t::NextSpikeTrajectory) = t.time_to_next_spike

#####################
# Simulator Methods #
#####################

# TODO: more general method with a way to give input spikes and a callback to receive output spikes

"""
    simulate_for_time_and_get_spikes(c::Component, s::State, t::Trajectory, ΔT)
    simulate_for_time_and_get_spikes(c::Component, s::State, ΔT)
    simulate_for_time_and_get_spikes(c::Component, ΔT)

Simulates the component for `ΔT` milliseconds and returns a vector giving all the output spikes which
occurred during the simulation.  The vector is ordered by time, and contains elements of the form
`(spiketime, (nestname, outputname))`, where `spiketime` is the time since the beginning of
the simulation at which this spike occurred, `outputname` is the name of a component output which spiked,
and `nestname` is as follows.
`nestname` will be `nothing` for outputs of `c`, `n1` for a subcomponent with name `n1`,
and `n1 => n2 => ... => n_n` for `c.subcomponents[n1].subcomponents[n2]....subcomponents[n_n]`.
"""
function simulate_for_time_and_get_spikes(c::Component, s::State, t::Trajectory, ΔT)
    spikes = []
    function f(itr, dt)
        for outspec in itr
            push!(spikes, (time_passed + dt, outspec))
        end
    end

    # states = State[s]
    # trajectories = Trajectory[t]
    time_passed = 0
    while time_passed < ΔT
        t = extend_trajectory(c, s, t)
        time_passed += trajectory_length(t)
        # push!(trajectories, t)

        if trajectory_length(t) == Inf
            break;
        end

        (s, t, _) = advance_time_by(c, s, t, trajectory_length(t), f)
        # push!(trajectories, t)
        # push!(states, s)
    end

    return spikes
end
simulate_for_time_and_get_spikes(c::Component, s::State, ΔT) = simulate_for_time_and_get_spikes(c, s, empty_trajectory(c), ΔT)
simulate_for_time_and_get_spikes(c::Component, ΔT) = simulate_for_time_and_get_spikes(c, initial_state(c), ΔT)

#################
# PoissonNeuron #
#################

next_spike(::PoissonNeuron, t::NextSpikeTrajectory) = :out

extend_trajectory(n::PoissonNeuron, s::OnOffState, ::EmptyTrajectory) = NextSpikeTrajectory(s.on ? exponential(n.rate) : Inf)
extend_trajectory(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory) = t

advance_time_by(::PoissonNeuron, s::OnOffState, t::NextSpikeTrajectory, ΔT) =
    let remaining_time = t.time_to_next_spike - ΔT
        if remaining_time == 0
            (s, EmptyTrajectory(), (:out,))
        elseif remaining_time > 0
            (s, NextSpikeTrajectory(remaining_time), ())
        else
            advancing_too_far_error(t, ΔT)
        end
    end

receive_input_spike(p::PoissonNeuron, s::OnOffState, ::Trajectory, inputname) =
    if inputname === :on
        (OnOffState(true), EmptyTrajectory(), ())
    elseif inputname === :off
        (OnOffState(false), EmptyTrajectory(), ())
    else
        error("Unrecognized input name: $inputname")
    end

#############
# Composite #
#############

mutable_version(t::Tuple) = collect(t)
mutable_version(n::NamedTuple) = Dict(pairs(n))
immutable_version(v::Vector) = Tuple(v)
immutable_version(d::Dict) = NamedTuple(d)

"""
    CompositeState <: State

State for a `CompositeComponent.`
"""
struct CompositeState{S} <: State
    substates::S
end

"""
    CompositeTrajectory <: Trajectory

Trajectory for a `CompositeComponent.`
"""
struct CompositeTrajectory{T} <: Trajectory
    subtrajectories::T
    trajectory_length::Float64
    has_next_spike::Bool
    next_spike_name::Union{Nothing, Symbol, Int} # TODO: should we have a `Name` constant rather than using `Symbol` and `Int`?
   
    CompositeTrajectory(subtrajectories::T, args...) where {
        T <: Union{
            Tuple{Vararg{<:Trajectory}},
            NamedTuple{<:Any, <:Tuple{Vararg{<:Trajectory}}}
        }
    } = new{T}(subtrajectories, args...)
end
function Base.show(io::IO, t::CompositeTrajectory)
    print(io, "CompositeTrajectory((")
    for (i, st) in enumerate(t.subtrajectories)
        print(io, st)
        if i != length(t.subtrajectories)
            print(io, ", ")
        end
    end
    print(io, "), $(t.trajectory_length), $(t.has_next_spike), $(t.next_spike_name))")
end

initial_state(c::CompositeComponent) = CompositeState(map(initial_state, c.subcomponents))
empty_trajectory(c::CompositeComponent) = CompositeTrajectory(map(empty_trajectory, c.subcomponents), 0.0, false, nothing)
trajectory_length(t::CompositeTrajectory) = t.trajectory_length
next_spike(::CompositeComponent, t::CompositeTrajectory) = t.has_next_spike ? t.next_spike_name : nothing

function first_pair_with_nonnothing_value(itr) # helper function
    for (name, v) in itr
        if v !== nothing
            return (name, v)
        end
    end
    return nothing
end
function extend_trajectory(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory)
    extended = map(extend_trajectory, c.subcomponents, s.substates, t.subtrajectories)
    times = map(trajectory_length, extended)
    mintime = minimum(times)
    at_min_time = (name for name in keys(c.subcomponents) if times[name] == mintime)
    outputted_spikes = Iterators.filter(
        ((compname, outname),) -> !isnothing(outname) && does_output(c, CompOut(compname, outname)),
        ((name, next_spike(c.subcomponents[name], extended[name])) for name in at_min_time)
    )
    has_output_spike = !isempty(outputted_spikes)

    outputname = if has_output_spike
        first(r for s in outputted_spikes for r in receivers(c, CompOut(s...)) if r isa Output).id
    else
        nothing
    end

    return CompositeTrajectory(extended, mintime, has_output_spike, outputname)
end

function nest_callback(f, nest_at)
    function nested(itr, ΔT)
        nested_itr = (
            (   isnothing(nst) ? nest_at : (nest_at => nst),
                name
            )
            for (nst, name) in itr
        )
        f(nested_itr, ΔT)
    end
end

# returns a collection of the same top-level type mapping `name => name`
names(t::Tuple) = Tuple(1:length(t))
names(n::NamedTuple) = NamedTuple(k=>k for k in keys(n))

# by the invariants, this:
# (1) does not extend time past where we have extended the trajectories to, and
# (2) does not extend time past the first spike which occurs in this component
function advance_time_by(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, ΔT, f::Function)
    @assert (trajectory_length(t) >= ΔT) "Should not advance a time past trajectory length!"
    advanced = map(
        (name, sc, ss, st) -> advance_time_by(sc, ss, st, ΔT, nest_callback(f, name)),
        names(c.subcomponents), c.subcomponents, s.substates, t.subtrajectories
    )
    advanced_states_and_trajs = mutable_version(map(((ss, st, _),) -> (ss, st), advanced))

    # send internal spikes to correct components; node the output spikes
    outspikes = []
    spike_queue = Queue{CompOut}()
    for (compname, (ss, st, spiking_out_names)) in pairs(advanced)
        for outname in spiking_out_names
            enqueue!(spike_queue, CompOut(compname, outname))
        end
    end
    handled_spikes = !isempty(spike_queue)

    while !isempty(spike_queue)
        output = dequeue!(spike_queue)
        subcomp = c.subcomponents[output.comp_name]

        for receiver in receivers(c, output)
            if receiver isa Output # note that this spike is output from `c`
                push!(outspikes, receiver.id)
            else # handle internal spiking
                (receivername, inname) = receiver.comp_name, receiver.in_name
                (substate, subtraj, new_out_spikes) = receive_input_spike(c.subcomponents[receivername], advanced_states_and_trajs[receivername]..., inname)
                advanced_states_and_trajs[receivername] = (substate, subtraj)

                for outname in new_out_spikes
                    enqueue!(spike_queue, (receivername, outname))
                end
            end
        end
    end

    return (
        CompositeState(immutable_version(map(((ss, _),) -> ss, advanced_states_and_trajs))),
        CompositeTrajectory(
            immutable_version(map(((_, st),) -> st, advanced_states_and_trajs)),
            trajectory_length(t) - ΔT,
            handled_spikes && t.has_next_spike,
            handled_spikes ? nothing : t.next_spike_name,
        ),
        outspikes
    )
end
advance_time_by(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, ΔT) = advance_time_by(c, s, t, ΔT, (_,_)->nothing)

function receive_input_spike(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, inname)
    outspikes = []
    spike_queue = Queue{Union{CompIn, Input}}()
    enqueue!(spike_queue, Input(inname))
    (s, t) = map(mutable_version, (s.substates, t.subtrajectories))

    while !isempty(spike_queue)
        in = dequeue!(spike_queue)
        for receiver in receivers(c, in)
            if receiver isa Output # handle output
                push!(outspikes, receiver.id)
            else # receiver isa CompIn   # receive to internal nodes
                cn = receiver.comp_name
                s[cn], t[cn], out_spike_names = receive_input_spike(c.subcomponents[cn], s[cn], t[cn], receiver.in_name)
                for outname in out_spike_names
                    for out_spike_receiver in receivers(c, outname)
                        enqueue!(spike_queue, out_spike_receiver)
                    end
                end
            end 
        end
    end

    return (
        CompositeState(immutable_version(s)),
        CompositeTrajectory(immutable_version(t), 0.0, false, nothing),
        outspikes
    )
end

end # module