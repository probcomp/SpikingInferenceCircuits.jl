"""
Remove JuliaNodes which have 0 inputs from the IR, and modify the IR
so that the values from these nodes are hardcoded constants.
"""
function inline_constant_nodes(ir::StaticIR)
    nodes_to_remove_values = Dict()
    for node in ir.nodes
        if hasproperty(node, :inputs) && length(node.inputs) == 0
            if !(node isa JuliaNode)
                error("Cannot remove 0-argument nodes which are not JuliaNodes; this one is a $(typeof(node)).")
            end

            nodes_to_remove_values[node] = node.function()
        end
    end

    return replace_nodes_with_constant_vals(ir, node_to_remove_values)
end
inline_constant_nodes(gf::StaticIRGenerativeFunction) =
    to_gf(
        inline_constant_nodes(get_ir(gf)),
        add_gf_name_suffix(gf, "constant_nodes_inlined")
    )

"""
Given a dict from node to the value that node should always output,
compiles the IR into a new IR where the constant nodes have been removed,
and the constant values compield into the receivers of the node outputs.
"""
function replace_nodes_with_constant_vals(ir::StaticIR, nodes_to_remove_values)
    name_to_new_node = Dict{Symbol, StaticIRNode}()
    builder = StaticIRBuilder()
    for node in ir.nodes
        if !haskey(nodes_to_remove_values, node) # remove the node if it is a key in this Dict
            new_node = node_without_deleted_as_parent(node, nodes_to_remove_values)
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

function node_without_deleted_as_parent(node, nodes_to_remove_values)
    if hasproperty(node, :inputs) && any(haskey(nodes_to_remove_values, n.name) for n in node.inputs)
        node_with_constant_inputs(
            node,
            Dict(
                n.name => nodes_to_remove_values[n.name]
                for n in node.inputs
                    if haskey(nodes_to_remove_values, n.name)
            )
        )
    else
        return node
    end
end

node_with_constant_inputs(node::JuliaNode, input_name_to_value::Dict) =
    JuliaNode(
        let delted_idx_values = node_input_idx_val_vec(node, input_name_to_value),
            nargs = length(node.inputs)
                (args...,) -> n.function(insert_values_at_indices(args, delted_idx_values, nargs)...)
        end,
        [n for n in node.inputs if !haskey(input_name_to_value, n)],
        node.name, node.typ
    )

node_with_constant_inputs(node::RandomChoiceNode, input_name_to_value::Dict) =
    RandomChoiceNode(
        with_constant_inputs_at_indices(
            node.dist,
            node_input_idx_val_vec(node, input_name_to_value)
        ),
        [n for n in node.inputs if !haskey(input_name_to_value, n)],
        node.addr, node.name, node.typ
    )
node_with_constant_inputs(node::GenerativeFunctionCallNode, input_name_to_value::Dict) =
    GenerativeFunctionCallNode(
        with_constant_inputs_at_indices(
            node.generative_function,
            node_input_idx_val_vec(node, input_name_to_value)
        ),
        [n for n in node.inputs if !haskey(input_name_to_value, n)],
        node.addr, node.name, node.typ
    )

# vector of `(idx, value)` pairs with the index and constant value
# for each input to this node with a constant value in the `input_name_to_value` dict
node_input_idx_val_vec(node, input_name_to_value) = [
        (i, input_name_to_value[n.name]) 
        for (i, n) in enumerate(node.inputs)
            if haskey(input_name_to_value, n.name)
    ]

# given a vector of "original arguments" and a vector of pairs (idx, val),
# and the total number of entries in each vector,
# returns a vector with the original arguments in order, and the given values
# at each given index
function insert_values_at_indices(original_args, val_idx_pairs, total_nargs)
    newargs = []
    og_idx = 1
    ins_idx = 1
    while length(newargs) < total_nargs
        (idx, val) = ins_idx > length(val_idx_pairs) ? (-1, -1) : val_idx_pairs[ins_idx]
        if length(newargs) == idx - 1
            push!(newargs, val)
            ins_idx += 1
        else
            push!(newargs, original_args[og_idx])
            og_idx += 1
        end
    end
    return newargs
end

### with_constant_inputs_at_indices ###

## For Distributions:
"""
Return a version of the given distribution/IR/GenFn for which the value
of certain arguments is fixed (and so fewer arguments are accepted).
`idx_val_pairs` is a vector of `(idx, val)` pairs giving an index
where the argument should have a constant value, and the constant value. 
""" 
with_constant_inputs_at_indices(cpt::CPT, idx_val_pairs) =
    CPT(cpt.dists[
        insert_values_at_indices(
            [Colon() for _=1:(num_inputs(cpt) - length(idx_val_pairs))],
            idx_val_pairs,
            num_inputs(cpt)
        )
    ])

with_constant_inputs_at_indices(lcpt::LabeledCPT{Ret}, idx_val_pairs) where {Ret} =
    LabeledCPT{Ret}(
        with_constant_inputs_at_indices(
            lcpt.cpt,
            [
                (idx, lcpt.input_values[idx](label))
                for (idx, label) in idx_val_pairs
            ]
        ),
        lcpt.output_values,
        let removed_indices = Set(idx for (idx, _) in idx_val_pairs)
            [
                bij for (i, bij) in enumerate(lcpt.input_values)
                if !(i in removed_indices)
            ]
        end
    )

## For StaticIR
with_constant_inputs_at_indices(ir::StaticIR, idx_val_pairs) = 
    replace_nodes_with_constant_vals(ir,
        Dict(
            ir.arg_nodes[idx] => val
            for (idx, val) in idx_val_pairs
        )
    )
with_constant_inputs_at_indices(gf::StaticIRGenerativeFunction, idx_val_pairs) =
    to_gf(
        with_constant_inputs_at_indices(get_ir(gf), idx_val_pairs),
        add_gf_name_suffix(gf, "with_constant_inputs_at_indices")
    )

## Implementation for Combinators in the `combinators/` directory