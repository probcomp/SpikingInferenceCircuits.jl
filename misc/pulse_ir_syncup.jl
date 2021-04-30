### Sketch of algorithm for aligning interfaces in the Pulse IR.

struct Interval
    min::Float64
    max::Float64
end
struct Window
    interval::Interval
    pre_hold::Float64 # do we know the pre_hold?  Or only track the post_hold?
    post_hold::Float64
end
# TODO: In the code below, I probably want to change from `Interval` to `Window`.

"""
Whether it is possible to specialize this component to the given input windows.
(If not, it must be because the inputs are too spread out.)
"""
can_support_inwindows(::Component, ::Dict{Input, Interval}) = error("Not implemented.")

"""
A specialized version of component which supports the given input windows.
"""
with_windows(::Component, ::Dict{Input, Interval}) = error("Not implemented.")

"""
For a spiking component specialized to some input windows,
return valid output windows.  (There may be multiple valid ways to position
the output windows if the input windows are shorter than the
largest windows this component can tolerate; this simply needs to return
a valid set of output windows.)
"""
output_windows(::Component) = error("Not implemented.")

can_support_inwindows(::Compositecomponent, input_windows) = true
function with_windows(circuit::CompositeComponent, input_windows)
    augmented_circuit = copy(circuit)
    windows = Dict{NodeName, Interval}(k => v for (k, v) in input_windows)
    for (name, subcomp) in topologically_ordered_subcomponents(circuit)
        in_windows = Dict(
            Input(name) => windows[CompIn(name, input)]
            for (input, _) in inputs(subcomp)
        )
        if can_support_inwindows(subcomp, in_windows)
            impl = with_inwindows(c, in_windows)
        else
            sync = SychronizationBuffer(in_windows)
            impl = with_windows(subcomp, out_windows(sync))
            insert_input_buffer!(augmented_circuit, name, sync)
        end
        for (o, window) in output_windows(impl)
            windows[o] = window
        end
        augmented_circuit[name] = impl
    end
    return augmented_circuit
end
output_windows(c::CompositeComponent) = 