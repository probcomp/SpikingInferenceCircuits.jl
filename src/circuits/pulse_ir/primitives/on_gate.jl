"""
    OnGate

Repeats spikes in the `:in` wire if a spike has been received in the `:on` wire recently.
"""
struct OnGate <: GenericComponent end
# TODO: implement non-async OnGate?

"""
    AsyncOnGate

In a memory window, outputs the same number of spikes as have been received in the `:in` wire so long
as an `:on` spike has been received within memory as well.
""" # TODO: clarify this docstring
struct AsyncOnGate <: GenericComponent end

# TODO: update this!  Add `Concrete` version!

Circuits.target(::Union{OnGate, AsyncOnGate}) = Spiking()
Circuits.inputs(::Union{OnGate, AsyncOnGate}) = NamedValues(:in => SpikeWire(), :on => SpikeWire())
Circuits.outputs(::Union{OnGate, AsyncOnGate}) = NamedValues(:out => SpikeWire())

"""
    ConcreteAsyncOnGate(ΔT, M, max_delay)

`AsyncOnGate` with concrete Pulse IR temporal interface.
"""
struct ConcreteAsyncOnGate <: ConcretePulseIRPrimitive
    ΔT::Float64 # time it remembers an off input
    max_delay::Float64
    M::Float64 # number of spikes which would have to be input to pass through without an `on` input
end
Circuits.abstract(::ConcreteAsyncOnGate) = OnGate()
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::ConcreteAsyncOnGate) = Circuits.$s(Circuits.abstract(g)))
end

async_on_gate(off::ConcreteOffGate) = ConcreteAsyncOnGate(off.ΔT, off.max_delay, off.M)

is_valid_input(g::ConcreteAsyncOnGate, d::Dict{Input, UInt}) = d[Input(:in)] < g.M

valid_strict_inwindows(g::ConcreteAsyncOnGate, d::Dict{Input, Window}) =
    (
       d[Input(:on)].pre_hold ≥ g.ΔT &&
       d[Input(:in)].pre_hold ≥ g.max_delay &&
       interval_length(d[Input(:in)]) ≤ g.ΔT - g.max_delay
    )
output_windows(g::ConcreteAsyncOnGate, d::Dict{Input, Window}) =
    Dict(Output(:out) => Window(
        Interval(
            d[Input(:in)].interval.min,
            d[Input(:in)].interval.max + g.max_delay
        ),
        0.0, 0.0 # TODO: if we have extra hold time in the inputs, we can probably do better
    ))