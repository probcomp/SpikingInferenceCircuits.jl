"""
    animation_to_frontend_format(initial_state, events)

Given the `initial_state` of a network and a list of events from the simulator,
produces an animation specification in the JSON format understood by the front-end.
May filter out states/events the front-end cannot render.

`initial_state` should be a `SpikingSimulator.state`; `events` should be a list of
`(time, compname, event::SpikingSimulator.Event)`).
"""
animation_to_frontend_format(initial_state, events) = Dict(
    "initial_states" => (
        if initial_state isa SpikingSimulator.CompositeState
            sts = collect(SpikingSimulator.pairs_deep(initial_state))
            filter(!isnothing, map(render_initial_state, sts))
        else
            []
        end
    ),
    "events" => filter(!isnothing, map(render_event, events))
)

render_event((time, compname, event)) = render_event(time, compname, event)
render_event(time, compname, s::SpikingSimulator.Spike) = Dict(
    "type" => "spike",
    "time" => time,
    "group" => convert_compname(compname),
    "out" => "$(s.outputname)"
)
render_event(time, compname, s::SpikingSimulator.StateChange) = 
    # Currently, the only states the front-end can only render OnOffStates
    if s.new_state isa SpikingSimulator.OnOffState
        Dict(
            "type" => "statechange",
            "time" => time,
            "group" => convert_compname(compname),
            "statetype" => "OnOffState",
            "val" => s.new_state.on,
        )
    else
        nothing
    end

render_initial_state((name, st)) = render_initial_state(name, st)
render_initial_state(_, ::SpikingSimulator.State) = nothing
render_initial_state(compname, s::SpikingSimulator.OnOffState) = Dict(
    "group" => convert_compname(compname),
    "statetype" => "OnOffState",
    "val" => s.on
)

convert_compname(::Nothing) = Union{Symbol, Integer}[]
convert_compname(name) = Union{Symbol, Integer}[name]
convert_compname((first, rest)::Pair) = prepend!(convert_compname(rest), [first])