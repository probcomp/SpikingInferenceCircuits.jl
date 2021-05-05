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
    pre_hold::Float64
    post_hold::Float64
    function Window(i::ClosedInterval, pre::Float64, post::Float64)
        @assert pre ≥ 0
        @assert post ≥ 0
        new(i, pre, post)
    end
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
                union(first.interval, window.interval),
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

#=
TODO -- clarify --
Is a hold on the output window the amount of time where we can guarantee no new output
spikes will arise so long as no new input spikes arise?
Or is it how long we can guarantee no output spikes will occur _even if_ input spikes arrive?

I think _probably_ we want it to be how long given that no new inputs occur.
=#

"""
Pulse IR primitive with concrete temporal interface (but not necessarily known failure probabilities
or a fixed implementation strategy.)
"""
abstract type ConcretePulseIRPrimitive <: GenericComponent end

"""
If a component has a concrete temporal interface, then
we can ask whether it supports a set of input windows,
and if so, ask for a corresponding output window.
"""
has_concrete_temporal_interface(::Component) = false
has_concrete_temporal_interface(::ConcretePulseIRPrimitive) = true
has_concrete_temporal_interface(c::CompositeComponent) = all(has_concrete_temporal_interface(sc) for sc in values(c.subcomponents))

# returns `CombinatoryInterface` or `SynchronousInterface` (or maybe `nothing`)
interface_type(::Component) = error("Not implemented.")

"""
An upper bound on the probability that a component fails to satisfy its interface when given valid inputs.
"""
# TODO: this failure probability can vary depending on the input windows and output windows.
# Could we have a function which reports the bound on a more case-by-case basis?
# Currently it has to report the bound for the worst-case input and output windows.
failure_probability_bound(::Component) = error("Not implemented.")

"""
Whether this spike-count assignment to the input lines is a valid input to a combinatory component.
"""
is_valid_input(::Component, ::Dict{Input, UInt}) = error("Not implemented.")
is_valid_input(c::ConcretePulseIRPrimitive, d::Dict{Input, UInt}) = is_valid_input(abstract(c), d)

# """
# Whether the component can support the given input windows.
# (TODO: what exactly does this mean?)
# """
# can_support_inwindows(::Component, ::Dict{Input, Window}) = error("Not implemented.")

"""
Whether the given input windows are valid for a _strict_ interface
for this component--ie. an interface where we can guarantee the distribution
of outputs in an output window based upon the inputs in the given input windows.
"""
valid_strict_inwindows(::Component, ::Dict{Input, Window}) = error("Not implemented.")
valid_strict_inwindows(c::ConcretePulseIRPrimitive, d::Dict{Input, Window}) = valid_strict_inwindows(abstract(c), d)

"""
The output windows for the composite component, given the windows in which inputs will arrive.
(If the input windows are valid strict windows, this will be the strict output;
if the input windows are not strict, these outputs will be broad enough to cover
all possible output windows for a strict input within the given loose input.)
"""
output_windows(::Component, input_windows::Dict{Input, Window})::Dict{Output, Window} = error("Not implemented.")
output_windows(c::ConcretePulseIRPrimitive, iw::Dict{Input, Window}) = output_windows(abstract(c), iw)

"""
The output windows for a synchronous component after a transition occurs.
"""
output_windows(::Component) = error("Not implemented.")

### Interface for a CompositeComponent ###
# TODO if we need it...
# interface_type(c::CompositeComponent) =
#     if !(all(has_concrete_temporal_interface(sc) for sc in values(c.subcomponents)))
#         nothing
#     else
#         length(Set(interface_type(sc)) for sc in c.subcomponents) == 1
#     end

failure_probability_bound(c::CompositeComponent) =
    1 - prod((1 - failure_probability_bound(sc)) for sc in c.subcomponents)

# This seems pretty hard to check for a general CompositeComponent.
is_valid_input(::CompositeComponent, ::Dict{Input, UInt}) = error("Not implemented.")

can_support_inwindows(c::CompositeComponent, d::Dict{Input, Window}) =
    !isnothing(output_windows(c, d))

# Windows for all the Inputs, CompIns, CompOuts, and Outputs which recieve inputs, given the input windows
# for the Inputs.
function nodename_windows(circuit::CompositeComponent, input_windows::Dict{Input, Window})
    windows = Dict{NodeName, Window}(k => v for (k, v) in input_windows)
    for (name, subcomp) in topologically_ordered_subcomponents(circuit)
        for input in keys_deep(inputs(subcomp))
            inwinds = (windows[inputter] for inputter in inputters(circuit, CompIn(name, input)))
            if !isempty(inwinds)
                windows[CompIn(name, input)] = containing_window(inwinds)
            end
        end

        in_windows = Dict{Input, Window}(
            Input(input) => windows[CompIn(name, input)]
            for input in keys_deep(inputs(subcomp))
        )

        for (o, window) in output_windows(subcomp, in_windows)
            windows[CompOut(name, Circuits.valname(o))] = window
        end
    end
    
    for outname in keys_deep(outputs(circuit))
        windows[Output(outname)] = containing_window(
            windows[compout]
            for compout in inputters(circuit, Output(outname))
        )
    end

    return windows
end

output_windows(c::CompositeComponent, input_windows::Dict{Input, Window}) =
    filter(nodename_windows(c, input_windows)) do (name, _); name isa Output; end;

valid_strict_inwindows(c::CompositeComponent, input_windows::Dict{Input, Window}) =
    let windows = nodename_windows(c, input_windows)
        all(
            valid_strict_inwindows(subcomponent, Dict(
                CompIn(subcomp_name, valname) => windows[CompIn(subcomp_name, valname)]
                for valname in keys(inputs(subcomp))
            ))
            for (subcomp_name, subcomp) in c.subcomponents
        )
    end

# TODO: other checks on the validity of a CompositeComponent
# e.g. can we check whether the set of outputs from a component are all
# valid inputs to a component it feeds into?