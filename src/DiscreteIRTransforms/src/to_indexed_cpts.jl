# TODO: For clarity, it would probably be better to have a specific `DomainBijection` type
# which is either `Identity`, `EnumeratedBijection`, or `ProductBijection`
# (corresponding to `nothing`, `::Bijection`, and `::Vector` in this code)

function to_indexed_cpts(ir::StaticIR, arg_domains)
    ir = to_labeled_cpts(ir, arg_domains)
    original_domains = get_domains(ir.nodes, arg_domains)

    new_domains = Dict()

    # for node with values `EmumeratedDomain(1:N)`, domain_bijections[node.name] is `nothing`
    # for node with any other EnumeratedDomain , domain_bijections[node.name] is a bijection from idx to label
    # for node with a ProductDomain, domain_bijections[node.name] is a vector of (vector | bijection | nothing) giving the mapping for the sub-domains.
    domain_bijections = Dict()

    for (name, old_domain) in original_domains
        (new_domains[name], domain_bijections[name]) = to_indexed_domain(old_domain)
    end

    name_to_new_node = Dict{Symbol, StaticIRNode}()
    builder = StaticIRBuilder()
    for node in ir.nodes
        new_node = to_indexed_cpt_node(node, original_domains, domain_bijections)
        new_node = update_inputs(new_node, name_to_new_node)
        name_to_new_node[new_node.name] = new_node

        add_node!(builder, new_node)
        if ir.return_node.name == new_node.name
            Gen.set_return_node!(builder, new_node)
        end
    end

    return (build_ir(builder), domain_bijections, domain_bijections[ir.return_node.name])
end

# TODO: we should ideally get `track_diffs` and `cache_julia_nodes` from the original `gf`!
function to_indexed_cpts(gf::StaticIRGenerativeFunction, arg_domains)
    (ir, bijs...) = to_indexed_cpts(get_ir(gf), arg_domains)
    gf = eval(Gen.generate_generative_function(
            ir,
            Symbol("$(typeof(gf))__indexed"); track_diffs=false, cache_julia_nodes=true
        ))
    @load_generated_functions()
    return (gf, bijs...)
end

### idx_to_label / label_to_idx utils ###

# Convert index to label or label to index
# from one of the bijections/nested-bijection vectors
# used to map Domains to indices

idx_to_label(::Nothing) = identity
idx_to_label(bij::Bijection) = idx -> bij[idx]
idx_to_label(bijs::Vector) =
    indices -> map.(map(idx_to_label, bijs), indices)

label_to_idx(::Nothing) = identity
label_to_idx(bij::Bijection) = l -> bij(l)
label_to_idx(bijs::Vector) =
    labels -> map.(map(label_to_idx, bijs), labels)

### to_indexed_domain ###
to_indexed_domain(old_domain::EnumeratedDomain) =
    (
        EnumeratedDomain(1:length(old_domain)),
        if old_domain == EnumeratedDomain(1:length(old_domain))
            nothing
        else
            Bijection(Dict(enumerate(old_domain)))
        end
    )
to_indexed_domain(old_domain::ProductDomain) =
    let (new_domains, sub_bijections) = unzip(map(to_indexed_domain, old_domain.sub_domains))
        (
            ProductDomain(new_domains),
            collect(sub_bijections)
        )
    end

### to_indexed_cpt_node ###
valtype_expr(::Nothing) = :Int
valtype_expr(::Bijection) = :Int
valtype_expr(::Vector{<:Union{<:Bijection, Nothing}}) = :(Vector{Int})
# TODO: I could be more detailed with the types rather than just returning generic `Vector` sometimes
valtype_expr(::Vector) = :Vector

to_indexed_cpt_node(node::ArgumentNode, _, bijections) = @set(node.typ = valtype_expr(bijections[node.name]))
function to_indexed_cpt_node(node::GenerativeFunctionCallNode, old_domains, bijections)
    new_gf, _, ret_bijection = to_indexed_cpts(
            node.generative_function,
            [old_domains[n.name] for n in node.inputs]
        )
    @assert bijections[node.name] == ret_bijection "Got: $(bijections[node.name]) != $ret_bijection"
    return setproperties(node, (generative_function = new_gf, typ=valtype_expr(bijections[node.name])))
end
dom_bij_vec(bij) = [bij[i] for i=1:length(bij)]
to_indexed_cpt_node(node::RandomChoiceNode, _, bijections) =
    if node.dist isa CPT
        node
    else
        @assert node.dist isa LabeledCPT
        @assert bijections[node.name] isa Union{Nothing, Bijection}
        setproperties(node, (dist=node.dist.cpt, typ=valtype_expr(bijections[node.name])))
    end

function to_indexed_cpt_node(node::JuliaNode, _, bijections)
    output_to_idx = label_to_idx(bijections[node.name])
    
    indexed_fn(indices...) =
        output_to_idx(
            node.fn(
                (
                    idx_to_label(bijections[p.name])(idx)
                    for (idx, p) in zip(indices, node.inputs)
                )...
            )
        )
    
    return setproperties(node, (fn = indexed_fn, typ=valtype_expr(bijections[node.name])))
end