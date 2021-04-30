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