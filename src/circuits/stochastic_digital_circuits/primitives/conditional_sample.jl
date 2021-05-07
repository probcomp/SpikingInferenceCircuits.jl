"""
    ConditionalSample(P::Matrix{Float64})

Unit to sample `y ~ P(Y ; x)` and output `1/P(y ; x)`.

`P` should be a matrix where `P[x, y]` is `P(y ; x)`.
"""
# TODO: change what `P` should be?  Perhaps to a `Vector` of `Categorical`s?
# Or perhaps switch what the rows vs columns mean?
struct ConditionalSample <: GenericComponent
    P::Matrix{Float64}
    function ConditionalSample(P::Matrix{Float64})
        @assert all(isapprox(x, 1.0) for x in sum(P, dims=2)) "∃ x s.t. ∑_y{P[y ; x]} ≂̸ 1.0"
        return new(P)
    end
end

in_domain_size(c::ConditionalSample) = size(c.P)[1]
out_domain_size(c::ConditionalSample) = size(c.P)[2]
prob_output_given_input(c::ConditionalSample, outval) = c.P[:,outval]

Circuits.inputs(c::ConditionalSample) =
    NamedValues(
        :in_val => FiniteDomainValue(in_domain_size(c))
    )

Circuits.outputs(c::ConditionalSample) =
    NamedValues(
        :value => FiniteDomainValue(out_domain_size(c)),
        :inverse_prob => NonnegativeReal()
    )