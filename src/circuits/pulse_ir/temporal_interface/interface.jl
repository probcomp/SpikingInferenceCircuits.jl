abstract type Interval end
struct ClosedInterval <: Interval
    min::Float64
    max::Float64
    function ClosedInterval(min, max)
        @assert min ≤ max
        return new(min, max)
    end
end
struct EmptyInterval <: Interval end
closed_or_empty_interval(a, b) = a ≤ b ? ClosedInterval(a, b) : EmptyInterval()
Interval(a, b) = closed_or_empty_interval(a, b)
interval_length(i::ClosedInterval) = i.max - i.min
interval_length(::EmptyInterval) = 0.
union(a::Interval, b::Interval) = ClosedInterval(min(a.min, b.min), max(a.max, b.max))
intersect(a::Interval, b::Interval) = 
    closed_or_empty_interval(max(a.min, b.min), min(a.max, b.max))

struct Window
    interval::ClosedInterval
    pre_hold::Float64 # do we know the pre_hold?  Or only track the post_hold?
    post_hold::Float64
end
start_of_pre_hold(w::Window) = w.interval.min - w.pre_hold
end_of_post_hold(w::Window) = w.interval.max + w.post_hold
interval_length(w::Window) = interval_length(w.interval)
pre_hold_interval(w::Window) = ClosedInterval(w.interval.min - w.pre_hold, w.interval.min)
post_hold_interval(w::Window) = ClosedInterval(w.interval.max, w.interval.max + w.post_hold)

"""
Given an iterator over Windows w₁, ..., wₙ, outputs a window W
such that if sᵢ is a spiketrain satisfying window wᵢ,
and S = ⋃sᵢ, then S satisfies W.
"""
containing_window(windows) = _containing_window(Iterators.peel(windows)...)
_containing_window(window, windows) =
    if isempty(windows)
        window
    else
        first, rest = Iterators.peel(windows)
        _containing_window(
            Window(
                union(w.interval, window.interval),
                interval_length(intersect(pre_hold_interval(window), pre_hold_interval(first))),
                interval_length(intersect(post_hold_interval(window), post_hold_interval(first)))
            ),
            rest
        )
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

# returns `CombinatoryInterface` or `SynchronousInterface` (or maybe `nothing`)
interface_type(::Component) = error("Not implemented.")

"""
An upper bound on the probability that a component fails to satisfy its interface when given valid inputs.
"""
failure_probability_bound(::Component) = error("Not implemented.")

"""
Whether this spike-count assignment to the input lines is a valid input to a combinatory component.
"""
is_valid_input(::Component, ::Dict{Input, UInt}) = error("Not implemented.")

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

### Interface for a CompositeComponent ###
has_concrete_temporal_interface(c::CompositeComponent) = !isnothing(interface_type(c))
interface_type(c::CompositeComponent) =
    if !(all(has_concrete_temporal_interface(sc) for sc in values(c.subcomponents)))
        nothing
    else
        length(Set(interface_type(sc)) for sc in c.subcomponents) == 1
    end

failure_probability_bound(c::CompositeComponent) =
    1 - prod((1 - failure_probability_bound(sc)) for sc in c.subcomponents)

# This seems pretty hard to check for a general CompositeComponent.
is_valid_input(::CompositeComponent, ::Dict{Input, UInt}) = error("Not implemented.")

can_support_inwindows(c::CompositeComponent, d::Dict{Input, Window}) =
    !isnothing(output_windows(c, d))

function output_windows(circuit::CompositeComponent, input_windows)
    windows = Dict{NodeName, Interval}(k => v for (k, v) in input_windows)
    for (name, subcomp) in topologically_ordered_subcomponents(circuit)
        in_windows = Dict(
            Input(input) => windows[CompIn(name, input)]
            for (input, _) in inputs(subcomp)
        )

        if can_support_inwindows(subcomp, in_windows)
            outwindows = output_windows(subcomp, in_windows)
        else
            return nothing
        end

        for (o, window) in output_windows
            windows[CompOut(name, valname(o))] = window
        end
    end

    return Dict(
        let outwindow = containing_window(
            windows[compout]
            for compout in inputters(circuit, Output(outname))
        )
            Output(outname) => outwindow
        end
        for outname in keys(outputs(circuit))
    )

    return filter(windows, ((k,_),) -> k isa Output)
end

# TODO: other checks on the validity of a CompositeComponent
# e.g. can we check whether the set of outputs from a component are all
# valid inputs to a component it feeds into?