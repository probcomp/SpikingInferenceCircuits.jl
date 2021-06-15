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
operation(::DeterministicGenFn{Generate}) = Generate(EmptySelection())

### implementation ###

# implementation for FiniteDomain output
determ_finite_domain_implementation(g::DeterministicGenFn) =
    RelabeledIOComponent(
        SDCs.MultiInputLookupTable(
            Tuple(d.n for d in g.input_domains),
            g.output_domain.n, g.fn
        ), (), (:out => :value,)
    ) # TODO: double check that this is right

# TODO: implementation for ProductDomain output
function determ_to_product_implementation(g::DeterministicGenFn, ::Spiking)
    @assert length(g.input_domains) == 1 "In the current implementation, a deterministic function outputting a ProductDomain value may only have a single value as input!"
    @assert all(d isa FiniteDomain for d in g.output_domain.subdomains) "Currently not implemented: outputting ProductDomain values where some subdomains are not FiniteDomains."
    input_domain = only(g.input_domains)

    return CompositeComponent(
        # we need to route specific values from one to the other, so we need to implement the I/O twice:
        # FiniteDomainValue --> SpikingCategoricalValue --> CompositeValue(SpikeWire() ...)
        implement(implement(inputs(g), Spiking(), :inputs), Spiking(), :inputs),
        implement(implement(outputs(g), Spiking(), :value), Spiking(), :value),
        (),
        (
            let outval = g.fn(inval)
                Input(:inputs => 1 => inval) => Output(:value => outidx => outval[outidx])
            end
            for inval=1:input_domain.n
                for outidx=1:length(g.output_domain.subdomains)
        ),
        g
    )
end

function Circuits.implement(g::DeterministicGenFn, t::Target)
    println("Implementing a deterministic gen fn...")
    impl = if g.output_domain isa FiniteDomain
            determ_finite_domain_implementation(g)
        elseif g.output_domain isa IndexedProductDomain
            determ_to_product_implementation(g, t)
        end
    println("Done implementing deterministic gen fn..")
    return impl
end

### Function --> DeterministicGenFn component ###

# Given a vector of values, infer what domain the values come from.
# If all values are vectors of the same length, assume this is an IndexedProductDomain;
# otherwise, assume it is a FiniteDomain.
infer_domain(vals) =
    if all(v isa Vector for v in vals) && length(Set(Iterators.map(length, vals))) == 1
        IndexedProductDomain(Tuple(infer_domain([v[i] for v in vals]) for i=1:length(first(vals))))
    else
        FiniteDomain(maximum(Set(vals)))
    end

# TODO: give a way for users to specify what the domains are, overriding the decision from `infer_domains`
gen_fn_circuit(f::Function, arg_domains::Tuple{Vararg{FiniteDomain}}, ::Op) where {Op <: GenFnOp} =
    DeterministicGenFn{Op}(
        arg_domains,
        Iterators.map(
            inputs -> f(inputs...)
            Iterators.product(map(vals, arg_domains)...)
        ) |> collect |> infer_domain,
        f
    )

gen_fn_circuit(::Function, ::Tuple{Vararg{<:Domain}}, ::Op) where {Op <: GenFnOp} =
    error("Currently, the circuit compiler supports deterministic functions where each input is from a FiniteDomain.")