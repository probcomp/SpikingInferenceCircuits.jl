struct Interval
    min::Float64
    max::Float64
end
struct Window
    interval::Interval
    pre_hold::Float64 # do we know the pre_hold?  Or only track the post_hold?
    post_hold::Float64
end

abstract type TemporalInterface end
struct CombinatoryInterface <: TemporalInterface
    in_windows::Dict{Input, Window}
    out_windows::Dict{Output, Window}
end
struct SynchronousInterface <: TemporalInterface
    out_windows::Dict{Output, Window}
    in_holds::Dict{Input, Interval}
end

"""
If a component has a concrete temporal interface, then
we can ask whether it supports a set of input windows,
and if so, ask for a corresponding output window.
"""
has_concrete_temporal_interface(::Component) = false

interface_type(::Component)::TemporalInterface = error("Not implemented.")

"""
Whether the component can support the given input windows.
"""
can_support_inwindows(::Component, ::Dict{Input, Window}) = error("Not implemented.")

"""
The output windows for the composite component, given the windows in which inputs will arrive.
"""
output_windows(::Component, input_windows::Dict{Input, Window})::Dict{Output, Window} = error("Not implemented.")

"""
The output windows for a synchronous component after a transition occurs.
"""
output_windows(::Component) = error("Not implemented.")

# draft! functions used in calls are not all implemented yet
function output_windows(circuit::CompositeComponent, input_windows)
    augmented_circuit = copy(circuit)
    windows = Dict{NodeName, Interval}(k => v for (k, v) in input_windows)
    for (name, subcomp) in topologically_ordered_subcomponents(circuit)
        in_windows = Dict(
            Input(input) => windows[CompIn(name, input)]
            for (input, _) in inputs(subcomp)
        )

        if can_support_inwindows(subcomp, in_windows)
            outwindows = output_windows(subcomp, in_windows)
        else
            sync = SychronizationBuffer(in_windows) # TODO
            outwindows = output_windows(subcomp, output_windows(sync))
            insert_input_buffer!(augmented_circuit, name, sync) # TODO
        end

        for (o, window) in output_windows
            windows[o] = window
        end
    end

    return filter(windows, ((k,_),) -> k isa Output)
end