struct LabeledCPT{Ret} <: Gen.Distribution{Ret}
    cpt::CPT
    output_values::Bijection{Int, Ret}
    input_values::Vector{<:Bijection{Int}}
end

"""
    LabeledCPT{Ret}(
        parent_domains::Vector,
        output_domain,
        assmt_to_probs
    )

A Labeled CPT taking `length(parent_domains)` inputs, with an output
domain given by the `output_domain` vector.
The input domain for parent `i` is given by a vector
of values `parent_domains[i]`.

`assmt_to_probs(assmt)` should be a probability vector
giving the probability of each element of `output_domain` (in
the same order as the `output_domain` vector), where `assmt`
is a tuple giving one value from each parent domain.
"""
function LabeledCPT{Ret}(
    parent_domains::Vector{<:Vector},
    output_domain::Vector,
    assmt_to_probs
) where {Ret}
    cartesian_parent_domains = CartesianIndices(Tuple(map(length, parent_domains)))

    assmts = (
        Tuple(
            parent_domains[i][v]
            for (i, v) in enumerate(Tuple(idx))
        )
        for idx in cartesian_parent_domains
    )
    probs = [assmt_to_probs(assmt) for assmt in assmts]

    return LabeledCPT(
        CPT(probs),
        Bijection(Dict{Int, Ret}(i => v for (i, v) in enumerate(output_domain))),
        [
            Bijection(Dict(i => v for (i, v) in enumerate(dom)))
            for dom in parent_domains
        ]
    )
end

_args_to_inds(l, args) = (bij(arg) for (bij, arg) in zip(l.input_values, args))
Gen.random(l::LabeledCPT, args...) = l.output_values[random(l.cpt, _args_to_inds(l, args)...)]
Gen.logpdf(l::LabeledCPT, val, args...) = logpdf(l.cpt, l.output_values(val), _args_to_inds(l, args)...)