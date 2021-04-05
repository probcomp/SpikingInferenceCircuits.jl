# Note: this currently only works when every input to every discrete distribution
# should be removed from the graph!  This is common when setting parameters to
# bernoulli, categorical, etc., but may be incorrect in other cases!
function to_labeled_cpts(ir::StaticIR, arg_domains)
    # if everything is already a CPT, we can just return the current IR
    if all(is_cpt(node.dist) for node in ir.choice_nodes)
        return ir
    end
    
    name_to_domain = get_domains(ir.nodes, arg_domains)
    names_to_delete = Set{Symbol}()
    to_replace = Dict{Symbol, StaticIRNode}()

    ### Label which nodes we need to replace / delete ###

    for node in ir.choice_nodes
        if !is_cpt(node.dist)
            to_replace[node.name] = get_labeled_cpt_node(node, name_to_domain)
            for parent in node.inputs
                push!(names_to_delete, parent.name)
            end
        end
    end
    for node in ir.call_nodes
        to_replace[node.name] = GenerativeFunctionCallNode(
            to_labeled_cpts(
                node.generative_function,
                Tuple(x -> name_to_domain[x.name] for x in node.inputs)
            ),
            node.inputs, node.addr, node.name, node.typ
        )
    end

    ### Build a new IR, using our REPLACE/DELETE labels ###

    name_to_new_node = Dict{Symbol, StaticIRNode}()
    builder = StaticIRBuilder()
    for node in ir.nodes
        if !(node.name in names_to_delete)
            new_node = haskey(to_replace, node.name) ? to_replace[node.name] : node
            new_node = update_inputs(new_node, name_to_new_node)
            name_to_new_node[new_node.name] = new_node

            add_node!(builder, new_node)
            if ir.return_node.name == new_node.name
                Gen.set_return_node!(builder, new_node)
            end
        end
    end

    return build_ir(builder)
end

# It's annoying that we have to do `eval` -- this is a TODO for the static IR
to_labeled_cpts(gf::StaticIRGenerativeFunction, arg_domains) = eval(
    Gen.generate_generative_function(
        to_labeled_cpts(Gen.get_ir(typeof(gf)), arg_domains),
        Symbol("$(typeof(foo))__labeled_cpts"); track_diffs=false, cache_julia_nodes=true
    ))

is_cpt(::CPT) = true
is_cpt(::LabeledCPT) = true
is_cpt(::Gen.Distribution) = false

# Get a RandomChoiceNode with a LabeledCPT distribution equivalent to `node`'s distribution,
# slurping in all the parents of the node (which are assumed to be JuliaNodes which produce
# probabilities parametrizing the discrete distribution)
function get_labeled_cpt_node(node::RandomChoiceNode, name_to_domain)
    @assert all(n isa JuliaNode for n in node.inputs) "Assumptions about restrictions on IR violated for node $(node.name)!"

    parent_inputs = collect(Iterators.flatten(pa.inputs for pa in node.inputs))
    parent_input_domains = Tuple(name_to_domain[x.name] for x in parent_inputs)
    input_domains = Tuple(name_to_domain[x.name] for x in node.inputs)
    cpt = get_labeled_cpt(
        node.dist, map(n -> n.fn, node.inputs),
        parent_input_domains, input_domains
    )
    return RandomChoiceNode(cpt, parent_inputs, node.addr, node.name, node.typ)
end

function _labeled_cpt(rettype, parent_domains, output_domain, parent_fns, fns_out_to_prob)
    assmt_to_val(assmt) = fns_out_to_prob(Tuple(fn(v) for (fn, v) in zip(parent_fns, assmt)))
    LabeledCPT{rettype}(collect(parent_domains), collect(output_domain), assmt_to_val)
end
get_labeled_cpt(d::Distribution, parent_fns, parent_domains, input_domains) =
    _labeled_cpt(
        Gen.get_return_type(d),
        parent_domains,
        get_domain(d, input_domains),
        parent_fns,
        assmt_to_probs(d)
    )