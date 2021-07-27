using DataStructures: DefaultDict

### Getting domains of nodes in generative functions ###

# get the domain of the return value
function get_ret_domain(s::StaticIRGenerativeFunction, arg_domains)
    ir = get_ir(s)
    return get_domains(ir.nodes, arg_domains)[ir.return_node.name]
end

# get the domain for each node in a `nodes` list from a Static IR
function get_domains(
    nodes, arg_domains;
    domain_type_constraints=Dict()
)
    domain_type_constraints = DefaultDict(() -> Domain, Dict{Symbol, Type{<:Domain}}(domain_type_constraints...))
    name_to_domain = Dict{Symbol, Domain}()

    # insert argument domains
    for (node, domain) in zip(nodes, arg_domains)
        name_to_domain[node.name] = domain
    end

    for node in nodes[(length(arg_domains) + 1):end]
        handle_node!(node, name_to_domain, domain_type_constraints[node.name])
    end

    return name_to_domain
end


## handle_node!(node::StaticIRNode, name_to_domain::Dict{Symbol, Domain}, domain_type_constraint::Type{<:Domain})
## add an entry to `name_to_domain` for this node, where the domain must be of the type `domain_type_constraint`

# handle_node!(::ArgumentNode, _, _) = nothing
function handle_node!(node::JuliaNode, name_to_domain, domain_type_constraint)
    input_domains = (name_to_domain[parent.name] for parent in node.inputs)
    assmts = Iterators.product(input_domains...)
    possible_domains = EnumeratedDomain(collect(unique(node.fn(assmt...) for assmt in assmts)))
    name_to_domain[node.name] = possible_outcomes_to_domain(
        possible_domains, domain_type_constraint
    )
end

function possible_outcomes_to_domain(possible_outcomes::EnumeratedDomain, domain_type_constraint=Domain)
    if domain_type_constraint == EnumeratedDomain
        possible_outcomes
    elseif domain_type_constraint == ProductDomain
        @assert valid_for_product_domain(possible_outcomes)
        to_product_domain(possible_outcomes)
    # else if no specific domain type is given (ie. domain_type_constraint == Domain)
    elseif valid_for_product_domain(possible_outcomes)
        to_product_domain(possible_outcomes)
    else
        possible_outcomes
    end
end

is_product_type(::AbstractArray) = true
is_product_type(_) = false
valid_for_product_domain(d::EnumeratedDomain) = (
    all(is_product_type(v) for v in vals(d)) &&
    let (first, rest) = Iterators.peel(vals(d))
        all(size(v) == size(first) for v in rest)
    end
)
to_product_domain(d::EnumeratedDomain) = ProductDomain(
        [
            map(x -> x[i], vals(d)) |> unique |> collect |> EnumeratedDomain |> possible_outcomes_to_domain
            for i=1:length(first(vals(d)))
        ]
    )

function handle_node!(node::RandomChoiceNode, name_to_domain, ::Type{Domain})
    domain = get_domain(node.dist, [name_to_domain[x.name] for x in node.inputs])
    name_to_domain[node.name] = domain
end
function handle_node!(node::GenerativeFunctionCallNode, name_to_domain, ::Type{Domain})
    name_to_domain[node.name] = get_ret_domain(node.generative_function, [name_to_domain[n.name] for n in node.inputs])
end
