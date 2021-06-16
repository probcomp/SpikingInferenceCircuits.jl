# Note: this currently only works when every input to every discrete distribution
# should be removed from the graph!  This is common when setting parameters to
# bernoulli, categorical, etc., but may be incorrect in other cases!
function to_labeled_cpts(ir::StaticIR, arg_domains)
    # if everything is already a CPT, we can just return the current IR
    if is_cpts(ir)
        return ir
    end

    parents_of_dists = Set{Symbol}()
    for node in ir.choice_nodes
        if !is_cpts(node.dist)
            for parent in node.inputs
                push!(parents_of_dists, parent.name)
            end
        end
    end

    name_to_domain = get_domains(
        ir.nodes, arg_domains;
        domain_type_constraints=Dict(p => EnumeratedDomain for p in parents_of_dists)
    )
    to_replace = Dict{Symbol, StaticIRNode}()

    for node in ir.choice_nodes
        if !is_cpts(node.dist)
            to_replace[node.name] = get_labeled_cpt_node(node, name_to_domain)
        end
    end
    for node in ir.call_nodes
        if !is_cpts(node.generative_function)
            to_replace[node.name] = GenerativeFunctionCallNode(
                to_labeled_cpts(
                    node.generative_function,
                    [name_to_domain[x.name] for x in node.inputs]
                ),
                node.inputs, node.addr, node.name, node.typ
            )
        end
    end

    ### Build a new IR, using our REPLACE/DELETE labels ###

    name_to_new_node = Dict{Symbol, StaticIRNode}()
    builder = StaticIRBuilder()
    for node in ir.nodes
        if !(node.name in parents_of_dists) # delete nodes which were direct arguments to a distribution node
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

to_labeled_cpts(gf::StaticIRGenerativeFunction, arg_domains) =
    to_gf(to_labeled_cpts(get_ir(gf), arg_domains), add_gf_name_suffix(gf, "labeled"))

# Get a RandomChoiceNode with a LabeledCPT distribution equivalent to `node`'s distribution,
# slurping in all the parents of the node (which are assumed to be JuliaNodes which produce
# probabilities parametrizing the discrete distribution)
function get_labeled_cpt_node(node::RandomChoiceNode, name_to_domain)
    @assert all(n isa JuliaNode for n in node.inputs) "Assumptions about restrictions on IR violated for node $(node.name)!"
    
    # determine an ordering of grandparent nodes to use
    grandparent_nodes = (parent.inputs for parent in node.inputs) |> Iterators.flatten |> unique |> collect
    gp_to_idx = Dict(gp => i for (i, gp) in enumerate(grandparent_nodes))

    grandparent_assmt_to_parent_assmt(gp_assmt) =
        [
            p.fn((
                gp_assmt[gp_to_idx[gp]]
                for gp in p.inputs
            )...)
            for p in node.inputs
        ]

    grandparent_assmt_to_probs(assmt) =
        let parent_assmt = grandparent_assmt_to_parent_assmt(assmt)
            assmt_to_probs(node.dist)(parent_assmt)
        end

    rettype = Gen.get_return_type(node.dist)
    grandparent_domains = [name_to_domain[x.name] for x in grandparent_nodes]
    output_domain = name_to_domain[node.name]

    lcpt = try
        LabeledCPT{rettype}(grandparent_domains, output_domain, grandparent_assmt_to_probs)
    catch e
        @error("Error while constructing labeled CPT for node $(node.name):")
        throw(e)
    end

    return RandomChoiceNode(lcpt, grandparent_nodes, node.addr, node.name, node.typ)
end