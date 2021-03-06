"""
    replace_return_node(gf::StaticIRGenerativeFunction)

Returns a Generative Function equivalent to `gf`, except with a different return value
(in particular, the return value will be the first node in the gen fn).
"""
replace_return_node(gf::StaticIRGenerativeFunction) =
    to_gf(replace_return_node(get_ir(gf)), add_gf_name_suffix(gf, "changedReturnValue"))
function replace_return_node(ir::StaticIR)
    keep_return_node = any(
        ir.return_node in node.inputs
        for node in Iterators.flatten(
            (ir.julia_nodes, ir.call_nodes, ir.choice_nodes)
        )
        # always keep nodes where there could be random choices!
    ) || !(ir.return_node isa JuliaNode)
    

    builder = StaticIRBuilder()
    for node in ir.nodes
        if keep_return_node || !(node == ir.return_node)
            add_node!(builder, node)
        end
    end
    Gen.set_return_node!(builder, first(Iterators.flatten((ir.arg_nodes, ir.choice_nodes, ir.call_nodes))))
    return build_ir(builder)
end