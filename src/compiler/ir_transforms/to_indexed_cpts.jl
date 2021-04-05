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

# TODO: we should ideally get `track_diffs` and `cache_julia_nodes` from the original `gf`!
function to_indexed_cpts(gf::StaticIRGenerativeFunction, arg_domains)
    (ir, bijs) = to_indexed_cpts(Gen.get_ir(typeof(gf)), arg_domains)
    return (
        eval(Gen.generate_generative_function(
            ir,
            Symbol("$(typeof(foo))__indexed"); track_diffs=false, cache_julia_nodes=true
        )),
        bijs
    )
end

dom_bij(vec) =
    if vec == collect(1:length(vec))
        nothing
    else
        Bijection(Dict([i => val for (i, val) in enumerate(vec)]))
    end

# Get a transformed node operating on indices instead of values,
# and a bijection specification for how the indices correspond to the values.
# The bijection will either be a `Bijection`, `nothing` (if no transformation is needed),
# or a dictionary from addresses to sub-bijection-specifications
_node_for_indexed_cpt(node::ArgumentNode, domains) = (@set(node.typ = :Int), dom_bij(domains[node.name]))
function _node_for_indexed_cpt(node::GenerativeFunctionCallNode, domains)
    new_gf, bij = to_indexed_cpts(node.generative_function, [domains[n.name] for n in node.inputs])
    return (
        setproperties(node, (generative_function = new_gf, typ=:Int)),
        bij
    )
end

_node_for_indexed_cpt(node::RandomChoiceNode, _) =
    if node.dist isa CPT
        (node, nothing)
    else
        (
            setproperties(node, (dist = node.dist.cpt, typ=:Int)),
            node.dist.output_values # **
        )
    end

function _node_for_indexed_cpt(node::JuliaNode, domains)
    out_domain = domains[node.name]

    in_domains = (domains[n.name] for n in node.inputs)
    inds_to_original_val = Dict( # **
        Tuple(idx) => node.fn(assmt...)
        for (idx, assmt) in zip(
            CartesianIndices(Tuple(1:length(dom) for dom in in_domains)),
            Iterators.product(in_domains...)
        )
    )
    
    og_val_to_idx = Dict(val => i for (i, val) in enumerate(out_domain))

    inds_to_idx = Dict(inds => og_val_to_idx[val] for (inds, val) in inds_to_original_val)

    indexed_fn(args...) = inds_to_idx[args]
    output_bijection = inv(Bijection(og_val_to_idx))

    return (
        setproperties(node, (fn = indexed_fn, typ=:Int)),
        output_bijection
    )
end