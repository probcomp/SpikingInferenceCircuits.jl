"""
    StreamSamples

Given `x`, streams samples of `y` from `P(Y ; x)`.

`P` should be a matrix where `P[x, y]` is `P(y ; x)`.
"""
struct StreamSamples <: GenericComponent
    P::Matrix{Float64}
end

in_domain_size(c::StreamSamples) = size(c.P)[1]
out_domain_size(c::StreamSamples) = size(c.P)[2]
prob_output_given_input(c::StreamSamples, outval) = c.P[:,outval]

Circuits.target(::StreamSamples) = Spiking()
Circuits.inputs(s::StreamSamples) = implement_deep(FiniteDomainValue(in_domain_size(s)), Spiking())
Circuits.outputs(s::StreamSamples) = implement_deep(FiniteDomainValue(out_domain_size(s)), Spiking())

# TODO (maybe): a somewhat richer interface for obtaining the I/O interface for a single output
# vs the interface for obtaining many outputs

"""
    ConcreteStreamSamples(ss:StreamSamples, ΔT, dist_on_num_samples)

Implementation of `StreamSamples` `ss` with a concrete Pulse IR interface.

For ΔT after receiving an input spike, will stream samples.  The distribution over the number
of samples emitted in time `t ≤ ΔT` since the input spike arrives is given by
dist_on_num_samples(t).  Each sample will be indicated by one output spike
in one of the output lines, and will be sampled from P(y ; x) (= `P[x, y]`).

Won't output anything if no input spike has been received recently.

Behavior is undefined for more than 1 input spike.
"""
struct ConcreteStreamSamples <: ConcretePulseIRPrimitive
    P::Matrix{Float64}
    ΔT::Float64
    dist_on_num_samples::Function
    # TODO: max delay?
end
ConcreteStreamSamples(ss::SampleStream, args...) = ConcreteSampleStream(ss.P, args...)
Circuits.abstract(ss::ConcreteStreamSamples) = StreamSamples(ss.P)
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::ConcreteStreamSamples) = Circuits.$s(Circuits.abstract(g)))
end

is_valid_input(ss::ConcreteStreamSamples, d::Dict{Input, UInt}) = d[Input(:in)] ≤ 1
valid_strict_inwindows(::ConcretePulseIRPrimitive, ::Dict{Input, Window}) = error("TODO")

# This gives the largest possible output windows, ie. the output windows in which
# the number of emitted samples is distributed according to dist_on_num_samples(ΔT).
# However, if the input windows are strict, any shorter output window after the input
# arrives will be valid, with a possibly different number of expected output spikes!
# TODO: is there a better way to deal with a component like this which can satisfy multiple
# output interfaces?
output_windows(ss::ConcreteSampleStream, d::Dict{Input, Window}) =
    let inwindow = containing_window(values(d))
        outwindow = Window(
        inwindow.interval, # TODO: add some delay!
        inwindow.pre_hold, Inf
    )
        Dict(Output(i) => outwindow)
    end