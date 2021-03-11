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

include("../../utils.jl")

using DataStructures: OrderedDict, Queue, enqueue!, dequeue!
# TODO: move the details of `PoissonNeuron` out of the simulator and into `targets/spiking/spiking.jl`
import ..Component, ..PrimitiveComponent, ..CompositeComponent, ..Spiking
import ..Output, ..Input, ..CompOut, ..CompIn, ..NodeName, ..receivers, ..does_output
const PrimComp = PrimitiveComponent{>:Spiking}

# TODO: could we do this better/in a more specific way?  This currently probably enables _some_ type checking but not performance improvements.
const Name = Union{Integer, Symbol, Pair}

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
Base.:(==)(a::OnOffState, b::OnOffState) = a.on == b.on

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

################################
# High-Level Simulator Methods #
################################

"""
    abstract type Event end

An event which can occur during a spiking simulation we may want to observe.

An `Event` occurs at some time for some component (but the time and component
are not part of the `Event` object).
"""
abstract type Event end

abstract type Spike <: Event end
struct InputSpike <: Spike
    name::Name
end
struct OutputSpike <: Spike
    name::Name
end

# """
#     Spike <: Event
#     Spike(outputname)

# The event that a spike was output by a component from the output with name `outputname`.
# """
# struct Spike <: Event
#     outputname::Name
# end

"""
    StateChange <: Event
    StateChange(new_state)

The event that a component's state changed to `new_state`.
"""
struct StateChange <: Event
    new_state::State
end

"""
    (new_state, new_traj, spiking_output_names) = advance_time_by(c::Component, ::State, ::Trajectory, ΔT, f::Function)

Same as `advance_time_by`, but calls `f(itr, dt)` at each `dt` at which an `Event` occurs
in the next `ΔT` seconds (including `ΔT`).  `itr` will be an iterator over pairs
`(compname, event)`, specifying each `event::Event` occurring for `c` or a subcomponent of `c`
specified by `compname` at time `dt`.  `compname` is a subcomponent name (possibly nested, or `nothing` to refer to `c`);
see documentation for `CompositeComponent`.

By default, for any component other than a `CompositeComponent`, it is assumed that there are no events occuring for internal components,
and so `f` is only called for events occurring at the top level to `c` (ie. output spikes from `c` and state changes for `c`).
"""
function advance_time_by(c::Component, s::State, t::Trajectory, ΔT, f::Function)
    (newstate, t, son) = advance_time_by(c, s, t, ΔT)
    f(
        Iterators.flatten((
            ((nothing, OutputSpike(name)) for name in son),
            newstate == s ? () : ((nothing, StateChange(newstate)),)
        )),
        ΔT
    )

    return (newstate, t, son)
end

# TODO: don't repeat so much from the `advance_time_by` above
"""
    (new_state, new_traj, spiking_output_names) = receive_input_spike(c::Component, s::State, t::Trajectory, inname, f::Function)

Same as `advance_time_by`, but calls `f(itr)` before returning, where `itr` is an iterator
over values `(compname, event)` specifying each `event::Event` occurring in `c` or a subcomponent
(specified by `compname`) when this input spike is received.  `compname` is a subcomponent name (possibly nested, or `nothing` to refer to `c`);
see documentation for `CompositeComponent`.

By default, for any component other than a `CompositeComponent`, it is assumed that there are no events occuring for internal components,
and so `f` is only called for events occurring at the top level to `c` (ie. output spikes from `c` and state changes for `c`).
"""
function receive_input_spike(c::Component, s::State, t::Trajectory, inname, f::Function)
    (newstate, t, son) = receive_input_spike(c, s, t, inname)
    f(
        Iterators.flatten((
            ((nothing, InputSpike(inname)),),
            ((nothing, OutputSpike(name)) for name in son),
            newstate == s ? () : ((nothing, StateChange(newstate)),)
        ))
    )

    return (newstate, t, son)
end


# TODO: more general method with a way to give input spikes and a callback to receive output spikes

"""
    simulate_for_time_and_get_events(c::Component, s::State, t::Trajectory, ΔT)
    simulate_for_time_and_get_events(c::Component, s::State, ΔT)
    simulate_for_time_and_get_events(c::Component, ΔT)

Simulates the component for `ΔT` milliseconds and returns a vector giving all the events which
occurred during the simulation.  The component must not have any recurrent connections to its own inputs.

The outputted vector is ordered by time, and contains elements of the form
`(time, compname, event)`, where `time` is the time since the beginning of
the simulation at which this event occurred, `compname` specifies the component for which
this event occurred, and `event` is the `Event` which occurred for this component at this time.

`compname` will be a subcomponent name (possibly nested, `nothing` to refer to `c`); see documentation
for `CompositeComponent`.
"""
function simulate_for_time_and_get_events(c::Component, s::State, t::Trajectory, ΔT; initial_inputs=())
    events = Tuple{Float64, Union{Nothing, Name}, Event}[]
    time_passed = 0

    # function to add events to the array
    function f(itr, dt)
        for (compname, event) in itr
            push!(events, (time_passed + dt, compname, event))
        end
    end

    for input in initial_inputs
        (s, t, output_names) = receive_input_spike(c, s, t, input, itr -> f(itr, 0.))
        f(Iterators.flatten((
            ((nothing, InputSpike(input)),),
            ((nothing, OutputSpike(n)) for n in output_names)
        )), 0.)
    end

    while time_passed < ΔT
        t = extend_trajectory(c, s, t)
        extending_by = trajectory_length(t)

        if trajectory_length(t) == Inf
            break;
        end

        (s, t, _) = advance_time_by(c, s, t, trajectory_length(t), f)
        time_passed += extending_by
    end

    return events
end
simulate_for_time_and_get_events(c::Component, s::State, ΔT; initial_inputs=()) =
    simulate_for_time_and_get_events(c, s, empty_trajectory(c), ΔT; initial_inputs)
simulate_for_time_and_get_events(c::Component, ΔT; initial_inputs=()) =
    simulate_for_time_and_get_events(c, initial_state(c), ΔT; initial_inputs)

simulate_for_time_and_get_spikes_and_primitive_statechanges(c, args...; kwargs...) =
    filter(simulate_for_time_and_get_events(c, args...; kwargs...)) do (_, compname, event)
        (event isa Spike) || (event isa StateChange && c[compname] isa PrimitiveComponent{>:Spiking})
    end

#############
# Composite #
#############

# TODO: I should probably just use the mutable versions everywhere; no need for immutability, except
# checking if the old state equals the current state when outputting `Event`s.
mutable_version(t::Tuple) = Vector{general_type(t)}(collect(t))
mutable_version(n::NamedTuple) = OrderedDict{Symbol, general_type(n)}(pairs(n))
immutable_version(v::Vector) = Tuple(v)
immutable_version(d::OrderedDict) = (;d...)

general_type(t) = _el_general_type(first(t))
_el_general_type(::State) = State
_el_general_type(::Trajectory) = Trajectory
_el_general_type(::Tuple{<:State, <:Trajectory}) = Tuple{<:State, <:Trajectory}

"""
    CompositeState <: State

State for a `CompositeComponent.`
"""
struct CompositeState <: State #{S} <: State
    substates #::S
    # CompositeState(s::T) where {T <: tup_or_namedtup(State)} = new{basetype(T)}(s)
end
pairs_deep(s::CompositeState) = Iterators.flatten((
    st isa CompositeState ? ((k => subkey, subst) for (subkey, subst) in pairs_deep(st)) : ((k, st),)
    for (k, st) in Base.pairs(s.substates)
))
Base.:(==)(a::CompositeState, b::CompositeState) = a.substates == b.substates

"""
    CompositeTrajectory <: Trajectory

Trajectory for a `CompositeComponent.`
"""
struct CompositeTrajectory <: Trajectory #{T} <: Trajectory
    subtrajectories #::T
    trajectory_length::Float64
    has_next_spike::Bool
    next_spike_name::Union{Nothing, Name}
   
    #CompositeTrajectory(st::T, args...) where {T <: tup_or_namedtup(Trajectory)} = new{basetype(T)}(st, args...)
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
    function nested(itr, dt...)
        nested_itr = (
            (   isnothing(name) ? nest_at : (nest_at => name),
                event
            )
            for (name, event) in itr
        )
        f(nested_itr, dt...)
    end
end

# returns a collection of the same top-level type mapping `name => name`
names(t::Tuple) = Tuple(1:length(t))
names(n::NamedTuple) = (;(k=>k for k in keys(n))...)

# by the invariants, this:
# (1) does not extend time past where we have extended the trajectories to, and
# (2) does not extend time past the first spike which occurs in this component
function advance_time_by(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, ΔT, f::Function)
    @assert (trajectory_length(t) >= ΔT) "Should not advance a time past trajectory length!"
    advanced = map(
        (name, sc, ss, st) -> advance_time_by(sc, ss, st, ΔT, nest_callback(f, name)),
        names(c.subcomponents), c.subcomponents, s.substates, t.subtrajectories
    )
    advanced_states = mutable_version(map(((ss, _, _),) -> ss, advanced))
    advanced_trajs = mutable_version(map(((_, st, _),) -> st, advanced))

    # advanced_states_and_trajs = mutable_version(map(((ss, st, _),) -> (ss, st), advanced))

    # send internal spikes to correct components; note the output spikes
    spikes_to_process = (CompOut(compname, outname) for (compname, (ss, st, spiking_out_names)) in pairs(advanced) for outname in spiking_out_names)
    handled_spikes = !isempty(spikes_to_process)

    (outspikes, _) = process_internal_spiking!(c, advanced_states, advanced_trajs, spikes_to_process, itr -> f(itr, ΔT))

    new_state = CompositeState(immutable_version(advanced_states))

    # callback
    f(Iterators.flatten((
        ((nothing, OutputSpike(name)) for name in outspikes),
        s == new_state ? () : ((nothing, StateChange(new_state)),)
    )), ΔT)

    return (
        new_state,
        CompositeTrajectory(
            # map(((_, st),) -> st, imm_advanced),
            immutable_version(advanced_trajs),
            trajectory_length(t) - ΔT,
            handled_spikes && t.has_next_spike,
            handled_spikes ? nothing : t.next_spike_name,
        ),
        outspikes
    )
end
advance_time_by(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, ΔT) = advance_time_by(c, s, t, ΔT, (_,_)->nothing)

function receive_input_spike(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, inname, f::Function)
    (s, t) = map(mutable_version, (s.substates, t.subtrajectories))

    (outspikes, state_changed) = process_internal_spiking!(c, s, t, (Input(inname),), f)

    new_state = CompositeState(immutable_version(s))

    # callback to note new events
    f(Iterators.flatten((
        ((nothing, InputSpike(inname)),),
        ((nothing, OutputSpike(name)) for name in outspikes),
        state_changed ? ((nothing, StateChange(new_state)),) : ()
    )))

    return (
        new_state,
        CompositeTrajectory(immutable_version(t), 0.0, false, nothing),
        outspikes
    )
end
receive_input_spike(c::CompositeComponent, s::CompositeState, t::CompositeTrajectory, inname) =
    receive_input_spike(c, s, t, inname, (_, _) -> nothing)

# TODO: document the interface for these functions
function process_internal_spiking!(
    c::CompositeComponent, s, t,
    initial_spikes, f
)
    spike_queue = Queue{NodeName}()
    for spike in initial_spikes
        enqueue!(spike_queue, spike)
    end

    outspikes = []
    state_changed = false

    while !isempty(spike_queue)
        spike = dequeue!(spike_queue)
        for receiver in receivers(c, spike)
            enqueue!(spike_queue, receiver)
        end

        sc = handle_spike!(c, s, t, spike, outspikes, spike_queue, f)
        state_changed = state_changed || sc
    end

    return (outspikes, state_changed)
end

function handle_spike!(_, _, _, r::Output, outspikes, _, _)
    push!(outspikes, r.id)
    return false
end
function handle_spike!(c, s, t, receiver::CompIn, outspikes, spike_queue, f)
    cn = receiver.comp_name
    oldstate = s[cn]
    s[cn], t[cn], out_spike_names = receive_input_spike(
        c.subcomponents[cn], s[cn], t[cn], receiver.in_name,
        nest_callback(f, cn)
    )
    for outname in out_spike_names
        enqueue!(spike_queue, CompOut(cn, outname))
    end

    return oldstate == s[cn]
end
handle_spike!(_, _, _, r::Union{Input, CompOut}, _, _, _) = false

end # module