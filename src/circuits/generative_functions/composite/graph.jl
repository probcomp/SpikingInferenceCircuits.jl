###############
# GenFn Graph #
###############
"""
    abstract type GenFnGraphNode{Op} end

A node in a graph forming a circuit to run `Op` for the `GenFn`.
An `InputNode` denotes an input value.
A `GenFnNode` is an internal `GenFn` operation.
"""
abstract type GenFnGraphNode{Op} end
struct GenFnNode{Op} <: GenFnGraphNode{Op}
    gen_fn::GenFn{Op}
    parents::Vector{Symbol}
end
struct InputNode{Op} <: GenFnGraphNode{Op}
    name::Symbol
end
output_domain(g::GenFnNode, _) = output_domain(g.gen_fn)
output_domain(i::InputNode, g) = input_domains(g)[i.name]

"""
    struct GraphGenFn{Op} <: GenFn{Op}

A GenFn component analogous to a static generative function:
it receives some number of input values, passes these values through
a graph of sub-generative functions, and ultimately outputs a value.

Any node which is traceable or outputs a probability must have an address.
"""
struct GraphGenFn{Op} <: CompositeGenFn{Op}
    input_domains::NamedTuple
    output_node_name::Symbol
    nodes::Dict{Symbol, GenFnGraphNode}
    addr_to_name::Dict{Symbol, Symbol}
    observed_addrs::Selection
    GraphGenFn{Propose}(ids, ons, n, a) = new{Propose}(ids, ons, n, a, EmptySelection())
    GraphGenFn{Generate}(ids, ons, n, a, oa::Selection) = new{Generate}(ids, ons, n, a, oa)
end
GraphGenFn(ids, ons, n, a, ::Propose) = GraphGenFn{Propose}(ids, ons, n, a)
GraphGenFn(ids, ons, n, a, op::Generate) = GraphGenFn{Generate}(ids, ons, n, a, op.observed_addrs)

input_domains(g::GraphGenFn) = g.input_domains
output_domain(g::GraphGenFn) = output_domain(g.nodes[g.output_node_name], g)
has_traceable_value(g::GraphGenFn) = any(has_traceable_value(g.nodes[name].gen_fn) for name in values(g.addr_to_name))
traceable_value(g::GraphGenFn) = NamedValues((
        addr => traceable_value(g.nodes[name].gen_fn)
        for (addr, name) in g.addr_to_name
    )...)
operation(g::GraphGenFn{Generate}) = Generate(g.observed_addrs)
arg_names(g::GraphGenFn) = keys(g.input_domains)

sub_gen_fns(g::GraphGenFn) = (;(name => node.gen_fn for (name, node) in g.nodes if node isa GenFnNode)...)
addr_to_name(g::GraphGenFn) = g.addr_to_name

# during `propose`, we score every traceable sub-gen-fn;
# during `generate`, we only score those traceable sub-gen-fns which we observe
# (We assume any node with an address is sampled from and thus has a prob output!)
score_outputter_names(g::GraphGenFn{Propose}) = values(g.addr_to_name)
score_outputter_names(g::GraphGenFn{Generate}) = (
    n for (a, n) in g.addr_to_name
    if !isempty(operation(g).observed_addrs[a])
)

# edges from argument values --> proposer input
arg_edges(g::GraphGenFn) = Iterators.flatten(
    arg_edges(g, name, node) for (name, node) in g.nodes
)
arg_edges(g::GraphGenFn, name, node::GenFnNode) = (
    arg_edge(g.nodes[parentname], parentname, comp_in_idx, name)
    for (comp_in_idx, parentname) in enumerate(node.parents)
)
arg_edges(::GraphGenFn, _, ::InputNode) = ()
arg_edge(parentnode::InputNode, _, comp_in_idx, gen_fn_name) =
    Input(:inputs => parentnode.name) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)
arg_edge(::GenFnNode, parentname, comp_in_idx, gen_fn_name) =
    CompOut(:sub_gen_fns => parentname, :value) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)

ret_edges(g::GraphGenFn) = ((CompOut(:sub_gen_fns => g.output_node_name, :value) => Output(:value)),)