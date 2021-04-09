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
struct GraphGenFn{Op} <: GenFn{Op}
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

# during `propose`, we score every traceable sub-gen-fn;
# during `generate`, we only score those traceable sub-gen-fns which we observe
# (We assume any node with an address is sampled from and thus has a prob output!)
prob_outputter_names(g::GraphGenFn{Propose}) = values(g.addr_to_name)
prob_outputter_names(g::GraphGenFn{Generate}) = (
    n for (a, n) in g.addr_to_name
    if !isempty(operation(g).observed_addrs[a])
)
num_internal_prob_outputs(g::GraphGenFn) = length(collect(prob_outputter_names(g)))

### Implement ###
Circuits.implement(g::GraphGenFn, ::Target) =
    CompositeComponent(
        inputs(g), outputs(g),
        (
            sub_gen_fns=sub_gen_fns_group(g),
            (let multgroup = multipliers_group(g)
                isempty(multgroup.subcomponents) ? () : (:multipliers => multgroup,)
            end)...
        ),
        Iterators.flatten((
            arg_value_edges(g),
            multiplier_edges(g),
            io_edges(g)
        )),
        g
    )

sub_gen_fns_group(g::GraphGenFn) = NamedComponentGroup(
        name => node.gen_fn for (name, node) in g.nodes if node isa GenFnNode
    )
# TODO: explore design tradeoffs between using pairwise multiplication vs multiple-input multiplication
multipliers_group(g) = IndexedComponentGroup(
        PositiveRealMultiplier(2) for _=1:(num_internal_prob_outputs(g) - 1)
    )

# edges from argument values --> proposer input
arg_value_edges(g::GraphGenFn) = Iterators.flatten(
    arg_value_edges(g, name, node) for (name, node) in g.nodes
)
arg_value_edges(g::GraphGenFn, name, node::GenFnNode) = (
    arg_value_edge(g.nodes[parentname], parentname, comp_in_idx, name)
    for (comp_in_idx, parentname) in enumerate(node.parents)
)
arg_value_edges(::GraphGenFn, _, ::InputNode) = ()
arg_value_edge(parentnode::InputNode, _, comp_in_idx, gen_fn_name) =
    Input(:inputs => parentnode.name) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)
arg_value_edge(::GenFnNode, parentname, comp_in_idx, gen_fn_name) =
    CompOut(:sub_gen_fns => parentname, :value) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)

# edges to perform pairwise multiplication of all the tracked probs
function multiplier_edges(g)
    outputters = collect(prob_outputter_names(g))
    if length(outputters) < 2
        return ()
    end

    firstname, rest = Iterators.peel(outputters)
    secondname, rest = Iterators.peel(rest)
    edges = Pair{<:CompOut, <:CompIn}[
        CompOut(:sub_gen_fns => firstname, :prob) => CompIn(:multipliers => 1, 1),
        CompOut(:sub_gen_fns => secondname, :prob) => CompIn(:multipliers => 1, 2)
    ]

    for (i, name) in zip(2:(num_internal_prob_outputs(g) - 1), rest)
        append!(edges, [
            CompOut(:multipliers => i - 1, :out) => CompIn(:multipliers => i, 1),
            CompOut(:sub_gen_fns => name, :prob) => CompIn(:multipliers => i, 2)
        ])
    end
    return edges
end

# input/output edges
io_edges(g::GraphGenFn) = Iterators.flatten((
        (
            (
                # probability output
                if has_prob_output(g)
                    if num_internal_prob_outputs(g) > 1
                        (CompOut(:multipliers => num_internal_prob_outputs(g) - 1, :out) => Output(:prob),)
                    else
                        (CompOut(:sub_gen_fns => first(prob_outputter_names(g)), :prob) => Output(:prob),)
                    end
                else
                    ()
                end
            )...,
            CompOut(:sub_gen_fns => g.output_node_name, :value) => Output(:value)
        ),
        trace_output_edges(g),
        obs_input_edges(g)
    ))
trace_output_edges(g::GraphGenFn) = (
        CompOut(:sub_gen_fns => g.addr_to_name[addr], :trace) => Output(:trace => addr)
        for addr in keys(trace_value(g))
    )
obs_input_edges(::GraphGenFn{Propose}) = ()
obs_input_edges(g::GraphGenFn{Generate}) = (
    Input(:obs => addr) => CompIn(:sub_gen_fns => g.addr_to_name[addr], :obs)
    for (addr, _) in g.addr_to_name if !isempty(operation(g).observed_addrs[addr])
)