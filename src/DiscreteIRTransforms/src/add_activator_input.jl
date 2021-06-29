function add_activator_input(ir::StaticIR, activator_input_name)
    @assert is_cpts(ir) "`add_activator_input` is currently only implemented for models using only CPTs."

    builder = StaticIRBuilder()
    activator_node = Gen.add_argument_node!(builder; name=activator_input_name)
    for node in ir.nodes
        is_return_node = node == ir.return_node
        if node isa JuliaNode || node isa GenerativeFunctionCallNode || node isa RandomChoiceNode
            if isempty(node.inputs)
                node = _add_activator_input(node, activator_node)
            end
        end

        add_node!(builder, node)

        if is_return_node
            set_return_node!(builder, node)
        end
    end

    return build_ir(builder)
end

add_activator_input(gf::StaticIRGenerativeFunction, activator_input_name) =
    to_gf(add_activator_input(get_ir(gf), activator_input_name), add_gf_name_suffix(gf, "withActivatorInput"))

# These can assume that there are currently 0 inputs:
_add_activator_input(node::JuliaNode, activator) = JuliaNode(
    x -> node.fn(), [activator], node.name, node.typ
)
_add_activator_input(node::RandomChoiceNode, activator) = RandomChoiceNode(
    add_activator_input(node.dist), [activator], node.addr, node.name, node.typ
)
_add_activator_input(node::GenerativeFunctionCallNode, activator) = GenerativeFunctionCallNode(
    add_activator_input(node.generative_function, gensym()),
    [activator], node.addr, node.name, node.typ
)

# This should be the only distribution we need to support (for now):
add_activator_input(cpt::CPTs.ZeroParentCPT) = CPT([cpt.dist])