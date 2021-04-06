function get_ret_domain(s::StaticIRGenerativeFunction, arg_domains)
    ir = get_ir(s)
    return get_domains(ir.nodes, arg_domains)[ir.return_node.name]
end
function get_ret_domain(s::Gen.Switch, arg_domains)
    # first domain should be branch selection
    @assert arg_domains[1] == 1:length(arg_domains[1])

    ret_domains = [get_ret_domain(branch, arg_domains[2:end]) for branch in s.branches]
    @assert length(unique(ret_domains)) == 1
    return first(ret_domains)
end

function get_domains(nodes, arg_domains)
    name_to_domain = Dict()

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
    name_to_domain[node.name] = [node.fn(assmt...) for assmt in assmts]
end
function handle_node!(node::RandomChoiceNode, name_to_domain)
    domain = get_domain(node.dist, [name_to_domain[x.name] for x in node.inputs])
    name_to_domain[node.name] = domain
end
function handle_node!(node::GenerativeFunctionCallNode, name_to_domain)
    name_to_domain[node.name] = get_ret_domain(node.generative_function, [name_to_domain[n.name] for n in node.inputs])
end