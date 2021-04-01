function to_indexed_cpts(ir::StaticIR, arg_domains)
    ir = to_labeled_cpts(ir, arg_domains)
    domains = get_domains(ir.nodes, arg_domains)

    domain_bijections = Dict()
    name_to_new_node = Dict{Symbol, StaticIRNode}()
    builder = StaticIRBuilder()
    for node in ir.nodes
        (new_node, bij) = _node_for_indexed_cpt(node, domains)

        new_node = update_inputs(new_node, name_to_new_node)
        name_to_new_node[new_node.name] = new_node
        
        if !isnothing(bij)
            domain_bijections[new_node.name] = bij
        end

        add_node!(builder, new_node)
        if ir.return_node.name == new_node.name
            Gen.set_return_node!(builder, new_node)
        end
    end

    return (build_ir(builder), domain_bijections)
end

dom_bij(vec) =
    if vec == collect(1:length(vec))
        nothing
    else
        Bijection(Dict([i => val for (i, val) in enumerate(vec)]))
    end

_node_for_indexed_cpt(node::ArgumentNode, domains) = (node, dom_bij(domains[node.name]))
function _node_for_indexed_cpt(node::GenerativeFunctionCallNode, domains)
    new_gf, bij = to_indexed_cpts(node.generative_function, [domains[n.name] for n in node.inputs])
    return (
        @set(node.generative_function = new_gf),
        bij
    )
end

_node_for_indexed_cpt(node::RandomChoiceNode, _) =
    if node.dist isa CPT
        (node, nothing)
    else
        (
            @set(node.dist = node.dist.cpt),
            node.dist.output_values
        )
    end

function _node_for_indexed_cpt(node::JuliaNode, domains)
    in_domains = (domains[n.name] for n in node.inputs)
    inds_to_original_val = Dict(
        Tuple(idx) => node.fn(assmt...)
        for (idx, assmt) in zip(
            CartesianIndices(Tuple(1:length(dom) for dom in in_domains)),
            Iterators.product(in_domains...)
        )
    )
    
    og_val_to_idx = Dict()
    for val in values(inds_to_original_val)
        if !haskey(og_val_to_idx, val)
            og_val_to_idx[val] = length(og_val_to_idx) + 1
        end
    end
    inds_to_idx = Dict(inds => og_val_to_idx[val] for (inds, val) in inds_to_original_val)

    indexed_fn(args...) = inds_to_idx[args]
    output_bijection = inv(Bijection(og_val_to_idx))

    return (
        @set(node.fn = indexed_fn),
        output_bijection
    )
end