update_inputs(node::ArgumentNode, _) = node
update_inputs(
    node::Union{JuliaNode, RandomChoiceNode, GenerativeFunctionCallNode},
    name_to_new_node
) = @set node.inputs = [name_to_new_node[n.name] for n in node.inputs]

function add_node!(builder, node)
    Gen._add_node!(builder, node)
    node_specific_add!(builder, node)
end
node_specific_add!(builder, node::ArgumentNode) = push!(builder.arg_nodes, node)
function node_specific_add!(builder, node::GenerativeFunctionCallNode)
    push!(builder.call_nodes, node)
    builder.addrs_to_call_nodes[node.addr] = node
end
function node_specific_add!(builder, node::RandomChoiceNode)
    push!(builder.choice_nodes, node)
    builder.addrs_to_choice_nodes[node.addr] = node
end
node_specific_add!(builder, node::JuliaNode) = push!(builder.julia_nodes, node)

### `get_ir` util ###
# to avoid the "world age" problem (https://discourse.julialang.org/t/world-age-problem-explanation/9714/3), 
# we need to use `Base.invokelatest`.
# `Gen.get_ir` only becomes defined when we create a new static Gen fn type,
# so if we create a new generative function _within_ our transformation functions, and
# we need to access its IR before we return to the top-level, the world age problem arises
get_ir(gf) = Base.invokelatest(Gen.get_ir, gf)

"""
Apply a transformation to a static GF's IR and return the resulting GF.

A suffix can be provided for the end of the transformed GF's name.
"""
# TODO: we should ideally get `track_diffs` and `cache_julia_nodes` from the original `gf`!
function gen_fn_for_ir_transformation(gf::StaticIRGenerativeFunction, transform_ir, name_ending="transformed")
    gf = eval(
        Gen.generate_generative_function(
            transform_ir(Gen.get_ir(typeof(gf))),
            Symbol("$(typeof(gf))__$name_ending"); track_diffs=false, cache_julia_nodes=true
        )
    )
    @load_generated_functions()
    return gf
end