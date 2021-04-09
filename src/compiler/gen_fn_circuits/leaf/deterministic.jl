#################
# Deterministic #
#################

#=
A deterministic function either:
1. Accepts FiniteDomainValues and outputs a FiniteDomainValue
or
2. Accepts FiniteDomainValues and outputs a ProductDomain values

Eventually, we could add support for taking ProductDomain values as inputs;
I think this will require logic to build circuits which include assignment buffers.
=#

"""
    DeterministicGenFn{Op} <: GenFn{Op}
    DeterministicDomainFn{Op}(input_domains::Tuple{Vararg{FiniteDomain}}, output_domain::Domain, fn::Function)

Circuit to perform an operation for a deterministic generative function (given by function `fn`, where the arguments
have the given input domain, and the emitted value is in the given output domain).

Currently, we only support having all input domains be `FiniteDomain`s; in the future we could extend
to support `ProductDomain`s.
"""
struct DeterministicGenFn{Op} <: GenFn{Op}
    input_domains::Tuple{Vararg{Domain}}
    output_domain::Domain
    fn::Function
end

# We could provide a default constructor for when the `output_domain` is not given as an argument.

input_domains(g::DeterministicGenFn) = g.input_domains
output_domain(g::DeterministicGenFn) = g.output_domain
has_traceable_value(::DeterministicGenFn) = false
# currently, I assume we never observe a deterministic node (since these are not traced)
operation(::DeterministicGenFn{Generate}) = Generate(Set())

### implementation ###

# CPT with deterministic outputs
deterministic_cpt(d::DeterministicGenFn) =
    CPT([
        onehot(d.output_domain.n, fn(input_vals...)::Int)
        for input_vals in Iterators.product((dom.n for dom in d.input_domains)...)
    ])
function onehot(n, i)
    @assert i <= n
    v = zeros(n)
    v[i] = 1
    return v
end

# implementation for FiniteDomain output
determ_finite_domain_implementation(g::DeterministicGenFn) =
    genfn_from_cpt_sample_score(
        CPTSampleScore(deterministic_cpt(g), true),
        g, false
    )

# TODO: implementation for ProductDomain output

Circuits.implement(g::DeterministicGenFn, ::Target) =
    if g.output_domain isa FiniteDomain
        determ_finite_domain_implementation(g)
    elseif g.output_domain isa IndexedProductDomain
        error("Not yet implemented: deterministic function from finite domains to product domain!")
    end

### Function --> DeterministicGenFn component ###

# Given a vector of values, infer what domain the values come from.
# If all values are vectors of the same length, assume this is an IndexedProductDomain;
# otherwise, assume it is a FiniteDomain.
infer_domain(vals) =
    if all(v isa Vector for v in vals) && length(Set(Iterators.map(length, vals))) == 1
        IndexedProductDomain(Tuple(infer_domain([v[i] for v in vals]) for i=1:length(first(vals))))
    else
        FiniteDomain(length(Set(vals)))
    end

# TODO: give a way for users to specify what the domains are, overriding the decision from `infer_domains`
gen_fn_circuit(f::Function, arg_domains::Tuple{Vararg{FiniteDomain}}, ::Op) where {Op <: GenFnOp} =
    DeterministicGenFn{Op}(
        arg_domains,
        Iterators.map(
            inputs -> f(inputs...),
            Iterators.product(map(vals, arg_domains)...)
        ) |> collect |> infer_domain,
        f
    )

gen_fn_circuit(::Function, ::Tuple{Vararg{<:Domain}}, ::Op) where {Op <: GenFnOp} =
    error("Currently, the circuit compiler supports deterministic functions where each input is from a FiniteDomain.")