"""
    ConditionalSampleScore(
        P::Matrix{Float64},
        draw_sample::Bool,
    )

Unit for sampling from a conditional distribution `P(Y | X)`, or observing
`y | x`, and outputting the score `P(y | x)`.

If `draw_sample` is true, this will sample a value; otherwise it will observe
a value.

`P` should be a matrix where `P[x, y]` is `P(y | x)`.
"""
# TODO: change what `P` should be?  Perhaps to a `Vector` of `Categorical`s?
# Or perhaps switch what the rows vs columns mean?
struct ConditionalSampleScore <: GenericComponent
    P::Matrix{Float64}
    sample::Bool
    function ConditionalSampleScore(P::Matrix{Float64}, s::Bool)
        @assert all(isapprox(x, 1.0) for x in sum(P, dims=2)) "∃ y s.t. ∑_y{P[y | x]} ≂̸ 1.0"
        return new(P, s)
    end
end

in_domain_size(c::ConditionalSampleScore) = size(c.P)[1]
out_domain_size(c::ConditionalSampleScore) = size(c.P)[2]
prob_output_given_input(c::ConditionalSampleScore, outval) = c.P[:,outval]

Circuits.inputs(c::ConditionalSampleScore) =
        NamedValues(
            :in_val => FiniteDomainValue(in_domain_size(c)),
            (c.sample ? () : (:obs => FiniteDomainValue(out_domain_size(c)),))...
        )

Circuits.outputs(c::ConditionalSampleScore) =
    NamedValues(
        (c.sample ? (:value => FiniteDomainValue(out_domain_size(c)),) : ())...,
        :prob => PositiveReal()
    )