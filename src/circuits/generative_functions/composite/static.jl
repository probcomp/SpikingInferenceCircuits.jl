##########################################
# Compilation Gen Static → GenFn Circuit #
##########################################

function gen_fn_circuit(ir::Gen.StaticIR, arg_domains::NamedTuple, op::Op) where {Op <: GenFnOp}
    @assert isempty(ir.trainable_param_nodes)
    @assert length(arg_domains) == length(ir.arg_nodes)

    nodes = Dict{Symbol, GenFnGraphNode{Op}}()
    addr_to_name = Dict{Symbol, Symbol}()
    domains = Dict{Symbol, Domain}()
    for node in ir.nodes
        handle_node!(nodes, node, domains, addr_to_name, arg_domains, op)
    end

    return GraphGenFn(
        arg_domains,
        ir.return_node.name,
        nodes,
        addr_to_name,
        op
    )
end
gen_fn_circuit(ir::Gen.StaticIR, arg_domains::Tuple, op::GenFnOp) =
    gen_fn_circuit(ir, (;(node.name => dom for (node, dom) in zip(ir.arg_nodes, arg_domains))...), op)

function handle_node!(nodes, node::Gen.ArgumentNode, domains, _, arg_domains, ::Op) where {Op <: GenFnOp}
    nodes[node.name] = InputNode{Op}(node.name)
    domains[node.name] = arg_domains[node.name]
end
function handle_node!(nodes, node::Gen.StaticIRNode, domains, addr_to_name, _, op::Op) where {Op <: GenFnOp}
    parent_names = Symbol[p.name for p in node.inputs]
    sub_gen_fn = gen_fn_circuit(node, parent_domains(parent_names, domains), static_ir_subop(node, op))
    
    nodes[node.name] = GenFnNode(sub_gen_fn, parent_names)
    domains[node.name] = output_domain(sub_gen_fn)
    if has_traceable_value(sub_gen_fn)
        addr_to_name[node.addr] = node.name
    end
end

# figure out the operation for a sub-generative-function, given the op for the top-level gen fn
static_ir_subop(_, ::Propose) = Propose()
static_ir_subop(::Gen.JuliaNode, ::Generate) = Generate(EmptySelection())
static_ir_subop(n::Gen.RandomChoiceNode, op::Generate) = Generate(n.addr in op.observed_addrs ? AllSelection() : EmptySelection())
static_ir_subop(n::Gen.GenerativeFunctionCallNode, op::Generate) = Generate(op.observed_addrs[n.addr])

parent_domains(parent_names, domains) = Tuple(domains[name] for name in parent_names)

# `gen_fn_circuit` for `StaticIRNode`s
gen_fn_circuit(gn::Gen.GenerativeFunctionCallNode, ads, op) =
    gen_fn_circuit(gn.generative_function, ads, op)
gen_fn_circuit(rcn::Gen.RandomChoiceNode, ads, op) = gen_fn_circuit(rcn.dist, ads, op)
gen_fn_circuit(jn::Gen.JuliaNode, ads, op) = gen_fn_circuit(jn.fn, ads, op)

# `gen_fn_circuit` for user-facing types
gen_fn_circuit(g::Gen.StaticIRGenerativeFunction, arg_domains, op) =
    gen_fn_circuit(Gen.get_ir(typeof(g)), arg_domains, op)