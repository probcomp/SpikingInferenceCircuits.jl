struct LabeledCPT{Ret} <: Gen.Distribution{Ret}
    cpt::CPT
    output_values::Bijection{Int, Ret}
    input_values::Vector{<:Bijection{Int}}

    # no Ret type parameter provided constructor
    LabeledCPT(cpt::CPT, outvals::Bijection{Int, Ret}, invals::Vector{<:Bijection{Int}}) where {Ret} = LabeledCPT{Ret}(cpt, outvals, invals)

    # constructor with some error checking to make sure domain sizes match CPT dimensions
    function LabeledCPT{Ret}(cpt::CPT, outvals::Bijection{Int, Ret}, invals::Vector{<:Bijection{Int}}) where {Ret}
        @assert length(outvals) == ncategories(cpt) "CPT size is not the same as the number of output values! ncategories(cpt) = $(ncategories(cpt)) but outvals = $outvals (length = $(length(outvals)))"
        for (i, (cpt_size, domain_bij)) in enumerate(zip(input_ncategories(cpt), invals))
            @assert cpt_size == length(domain_bij) "The CPT input size does not match the domain bijection along the $(i)th input dimension!  CPT input dimension size: $cpt_size; Domain size: $(length(domain_bij))"
        end

        new{Ret}(cpt, outvals, invals)
    end
end

parentless_labeled_cpt(Ret, output_domain, assmt_to_probs) =
    LabeledCPT{Ret}(CPT(assmt_to_probs(())), Bijection(Dict{Int, Ret}(i => v for (i, v) in enumerate(output_domain))), Bijection{Int, Any}[])

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
    parent_domains::Vector,
    output_domain,
    assmt_to_probs
) where {Ret}
    if length(parent_domains) == 0
        return parentless_labeled_cpt(Ret, output_domain, assmt_to_probs)
    end

    # cartesian_parent_domains = CartesianIndices(Tuple(map(length, parent_domains)))

    assmts = Iterators.product(parent_domains...)

    # assmts = (
    #     Tuple(
    #         parent_domains[i][v]
    #         for (i, v) in enumerate(Tuple(idx))
    #     )
    #     for idx in cartesian_parent_domains
    # )
    probs = [assmt_to_probs(assmt) for assmt in assmts]
    @assert all(Distributions.isprobvec(v) for v in probs) "Not all probvecs are actually probvecs!"

    return LabeledCPT(
        CPT(probs),
        Bijection(Dict{Int, Ret}(i => v for (i, v) in enumerate(output_domain))),
        [
            Bijection(Dict(i => v for (i, v) in enumerate(dom)))
            for dom in parent_domains
        ]
    )
end

function get_inv_with_err(bij, val)
    @assert val in image(bij) "Value $val should be in the image of bijection $(collect(bij))]"
    return bij(val)
end

_args_to_inds(l, args) = (
    get_inv_with_err(bij, arg)
    for (bij, arg) in zip(l.input_values, args)
)
function Gen.random(l::LabeledCPT, args...)
    inds = _args_to_inds(l, args)

    sampled_idx = random(l.cpt, inds...)
    if !(sampled_idx in domain(l.output_values))
        println("$sampled_idx not a key for $(l.output_values)!")
    end
    l.output_values[sampled_idx]
end
Gen.logpdf(l::LabeledCPT, val, args...) = logpdf(l.cpt, l.output_values(val), _args_to_inds(l, args)...)