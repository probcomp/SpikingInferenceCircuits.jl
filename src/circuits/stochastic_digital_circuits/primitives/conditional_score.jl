"""
    ConditionalScore(P::Matrix{Float64})

Unit to return `P(y ; x)` given `x` and `y`.

`P` should be a matrix where `P[x, y]` is `P(y ; x)`.
"""
# TODO: change what `P` should be?  Perhaps to a `Vector` of `Categorical`s?
# Or perhaps switch what the rows vs columns mean?
struct ConditionalScore <: GenericComponent
    P::Matrix{Float64}
    function ConditionalScore(P::Matrix{Float64})
        @assert all(isapprox(x, 1.0) for x in sum(P, dims=2)) "∃ x s.t. ∑_y{P[y | x]} ≂̸ 1.0"
        return new(P)
    end
end

in_domain_size(c::ConditionalScore) = size(c.P)[1]
out_domain_size(c::ConditionalScore) = size(c.P)[2]
prob_output_given_input(c::ConditionalScore, outval) = c.P[:,outval]

Circuits.inputs(c::ConditionalScore) =
        NamedValues(
            :in_val => FiniteDomainValue(in_domain_size(c)),
            :obs => FiniteDomainValue(out_domain_size(c))
        )

Circuits.outputs(::ConditionalScore) =
    NamedValues(
        :prob => PositiveReal()
    )