### Domain type ###
abstract type Domain end
"""Domain represented as a list of possible values."""
struct EnumeratedDomain{V} <: Domain
    vals::V
end
vals(d::EnumeratedDomain) = d.vals

"""
Product set of all tuples/vectors containing one object each subdomain.
(`vector_valued` is true if the domain contains vectors, false otherwise.)
"""
struct ProductDomain{T <: Tuple{Vararg{<:Domain}}} <: Domain
    sub_domains::T
    vector_valued::Bool
end
vals(d::ProductDomain) = (
        d.vector_valued ? collect(v) : v
        for v in Iterators.product(d.sub_domains)
    )

Base.iterate(d::Domain) = Base.iterate(vals(d))
Base.iterate(d::Domain, s) = Base.iterate(vals(d), s)
Base.length(d::Domain) = length(vals(d))
Base.:(==)(a::Domain, b::Domain) = all(x == y for (x, y) in zip(a, b))
Base.hash(d::Domain, h::UInt) = hash(collect(vals(d)), h)

### Getting domains of nodes in generative functions ###

# get the domain of the return value
function get_ret_domain(s::StaticIRGenerativeFunction, arg_domains)
    ir = get_ir(s)
    return get_domains(ir.nodes, arg_domains)[ir.return_node.name]
end
function get_ret_domain(s::Gen.Switch, arg_domains)
    # first domain should be branch selection
    @assert arg_domains[1] == EnumeratedDomain(1:length(arg_domains[1]))

    ret_domains = [get_ret_domain(branch, arg_domains[2:end]) for branch in s.branches]
    @assert all(d == first(ret_domains) for d in ret_domains[2:end]) "Different branches had different return domains! Domains for each branch: $ret_domains"
    return first(ret_domains)
end

# get the domain for each node in a `nodes` list from a Static IR
function get_domains(nodes, arg_domains)
    name_to_domain = Dict{Symbol, Domain}()

    # insert argument domains
    for (node, domain) in zip(nodes, arg_domains)
        name_to_domain[node.name] = domain
    end

    for node in nodes[(length(arg_domains) + 1):end]
        handle_node!(node, name_to_domain)
    end

    return name_to_domain
end

# handle_node!(::ArgumentNode, _) = nothing
function handle_node!(node::JuliaNode, name_to_domain)
    input_domains = (name_to_domain[parent.name] for parent in node.inputs)
    assmts = Iterators.product(input_domains...)

    first_assmt = first(assmts)
    first_val = node.fn(first_assmt...)
    # if first_val isa SeparateValueVector
    #     name_to_domain[node.name] = ProductDomain(Tuple(input_domains), true)
    # else
    name_to_domain[node.name] = EnumeratedDomain([node.fn(assmt...) for assmt in assmts])
    # end
end
function handle_node!(node::RandomChoiceNode, name_to_domain)
    domain = get_domain(node.dist, [name_to_domain[x.name] for x in node.inputs])
    name_to_domain[node.name] = domain
end
function handle_node!(node::GenerativeFunctionCallNode, name_to_domain)
    name_to_domain[node.name] = get_ret_domain(node.generative_function, [name_to_domain[n.name] for n in node.inputs])
end