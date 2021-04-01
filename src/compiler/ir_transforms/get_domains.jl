function get_domains(nodes, arg_domains)
    name_to_domain = Dict{Symbol, Vector}()

    # insert argument domains
    for (node, domain) in zip(nodes, arg_domains)
        name_to_domain[node.name] = domain
    end

    for node in nodes[(length(arg_domains) + 1):end]
        handle_node!(node, name_to_domain)
    end

    display(name_to_domain)
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