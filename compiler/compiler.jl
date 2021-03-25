using Gen
using Circuits

############
# Abstract #
############

"""
    abstract type Propose <: GenericComponent end

Component which implements the `Propose` operation
to return a trace and the probability of sampling that trace.
Outputs a return `:value`, the `:trace`, and the `:prob`.
"""
abstract type Propose <: GenericComponent end

"""
    input_domain_sizes(::Propose)

A `Tuple` or `NamedTuple` giving the sizes of the domains
for each input value to this `Propose`.
"""
input_domain_sizes(::Propose) = error("Not implemented.")
"""
    output_domain_size(::Propose)::Int

The size of the output value of this `Propose`.
"""
output_domain_size(::Propose)::Int = error("Not implemented.")

"""
    trace_value(::Propose)

The `Value` which is the `:trace` output for this `Propose`.
"""
trace_value(::Propose) = error("Not implemented.")

"""
    has_trace(::Propose)

Whether the `Propose` outputs a trace (or just a value).
(E.g. deterministic `Propose`s don't output a trace.)
"""
has_trace(::Propose) = error("Not implemented.")

Circuits.inputs(p::Propose) = CompositeValue(
        map(FiniteDomainValue, input_domain_sizes(p))
    )
Circuits.outputs(p::Propose) = NamedValues(
        :value => FiniteDomainValue(output_domain_size(p)),
        (has_trace(p) ? (:trace => trace_value(p),) : ())...,
        :prob => PositiveReal()
    )

#################
# Deterministic #
#################

"""
    DeterministicPropose <: Propose
    DeterministicPropose(input_domain_sizes::Tuple, fn::Function)

`Propose` for a deterministic function `fn` accepting `length(input_domain_sizes)`
arguments, from the sets
`{1, ..., input_domain_sizes[1]}`, ..., `{1, ..., input_domain_sizes[end]}`.
The output value is expected to be an integer from a set `{1, ..., out_domain_size}`.
"""
struct DeterministicPropose <: Propose
    input_domain_sizes::Tuple
    output_domain_size::Int
    fn::Function
end
function DeterministicPropose(input_domain_sizes::Tuple, fn::Function)
    output_domain_size = length(unique(fn(vals...) for vals in Iterators.product(input_var_domain_sizes)))
    DeterministicPropose(input_domain_sizes, output_domain_size, fn)
end
input_domain_sizes(p::DeterministicPropose) = p.input_domain_sizes
output_domain_size(p::DeterministicPropose) = p.output_domain_size
has_trace(::DeterministicPropose) = false

# CPT with deterministic outputs
function deterministic_cpt(d::DeterministicPropose)
    output_vals = Set(fn(vals...) for vals in Iterators.product(input_var_domain_sizes(d)))
    n_o = length(output_vals)
    sorted_vals = sort(collect(output_vals))
    val_to_idx = Dict(val => idx for (idx, val) in enumerate(sorted_vals))
    CPT([onehot(n_o, val_to_idx(fn(Tuple(vals)...))) for vals in CartesianIndices(input_var_domain_sizes(d))])
end
function onehot(n, i)
    v = zeros(n)
    v[i] = 1
    return v
end
Circuits.implement(p::DeterministicPropose, ::Target) =
    propose_from_cpt_sample_score(
        CPTSampleScore(deterministic_cpt(p), true),
        p, false
    )

"""
    propose_from_cpt_sample_score(cpt_ss::CPTSampleScore, p::Propose, val_to_trace::Bool)

A `CompositeComponent` which implements `p` using the given `cpt_ss`.  If `val_to_trace` is true,
the value output from `cpt_ss` is both output as `:value` and `:trace`; if this is false, it is only
output as `:value`.
"""
propose_from_cpt_sample_score(cpt_ss, p, val_to_trace) = CompositeComponent(
        inputs(p), outputs(p),
        (cpt_sample_score=cpt_ss,),
        Iterators.flatten((
            (Input(i) => CompIn(:cpt_sample_score, :in_vals => i) for i=1:length(inputs(p))),
            (
                CompOut(:cpt_sample_score, :value) => Output(:value),
                CompOut(:cpt_sample_score, :prob) => Output(:prob),
                (val_to_trace ? (CompOut(:cpt_sample_score, :value) => Output(:trace),) : ())...
            )
        )),
        p
    )

################
# Distribution #
################

struct DistributionPropose <: Propose
    cpt::CPT
end
input_domain_sizes(d::DistributionPropose) = input_ncategories(d.cpt)
output_domain_size(d::DistributionPropose) = ncategories(d.cpt)
has_trace(d::DistributionPropose) = true
trace_value(d::DistributionPropose) = FiniteDomainValue(output_domain_size(d))
Circuits.implement(d::DistributionPropose, ::Target) =
    propose_from_cpt_sample_score(CPTSampleScore(d.cpt, true), d, true)


#####################
# Graph of Proposes #
#####################

"""
    abstract type ProposeGraphNode end

A node in a graph of `Propose` operations.
An `InputNode` denotes an input value.
A `ProposeNode` is an internal `Propose` operation.
"""
abstract type ProposeGraphNode end
struct ProposeNode <: ProposeGraphNode
    propose::Propose
    parents::Vector{Symbol}
end
struct InputNode <: ProposeGraphNode
    name::Symbol
end
output_domain_size(p::ProposeNode, _) = output_domain_size(p.propose)
output_domain_size(i::InputNode, p) = input_domain_size(p)[i.name]

"""
    GraphPropose <: Propose

A `Propose` operation implemented via other "sub-Propose" operations.
(Eg. `Propose` for a static generative function which calls distributions
or other static generative functions.)

`nodes` should be a dictionary from node `name` to a `ProposeGraphNode`.
`input_domain_sizes` should be a properly-ordered NamedTuple mapping the names of the inputs
to their domain sizes.
`output_node_name` should be the name of the node whose value should be output.
(Currently, there must be an output value.)

`addr_to_ame` is a dictionary which specifies which nodes contribute to the `:trace` and `:prob` output.
For each sub-node with name `name whose value should be in the trace at address `addr`,
a mapping `addr => name` should be in the dictionary,
specifying that the trace for this subnode should appear at `:trace => :addr =>` in the output.
This also specifies that the probability of this node should be factored into the output `:prob`.
(Untraced values do not affect the `:prob`.)
"""
struct GraphPropose <: Propose
    input_domain_sizes::NamedTuple
    output_node_name::Symbol
    nodes::Dict{Symbol, ProposeGraphNode}
    addr_to_name::Dict{Symbol, Symbol}
end
input_domain_sizes(p::GraphPropose) = p.input_domain_sizes
output_domain_size(p::GraphPropose) = output_domain_size(p.nodes[p.output_node_name], p)
has_trace(p::GraphPropose) = true
trace_value(p::GraphPropose) = NamedValues((
        addr => trace_value(p.nodes[name].propose)
        for (addr, name) in p.addr_to_name
    )...)

num_proposers(p::GraphPropose) = length(p.addr_to_name)

### Implement ###
Circuits.implement(p::GraphPropose, ::Target) =
    CompositeComponent(
        inputs(p), outputs(p),
        (
            proposers=proposers_group(p),
            multipliers=multipliers_group(p)
        ),
        Iterators.flatten((
            arg_value_edges(p),
            multiplier_edges(p),
            io_edges(p)
        )),
        p
    )

proposers_group(p::GraphPropose) = NamedComponentGroup(
        name => node.propose for (name, node) in p.nodes if node isa ProposeNode
    )
multipliers_group(p::GraphPropose) = IndexedComponentGroup(
        PositiveRealMultiplier(2) for _=1:(num_proposers(p) - 1)
    )

# edges from argument values --> proposer input
arg_value_edges(p::GraphPropose) = Iterators.flatten(
        arg_value_edges(p, name, node) for (name, node) in p.nodes
    )
arg_value_edges(p::GraphPropose, name, node::ProposeNode) = (
        arg_value_edge(p.nodes[parentname], parentname, comp_in_idx, name)
        for (comp_in_idx, (parentname, domain_size)) in enumerate(
            zip(node.parents, input_domain_sizes(node.propose))
        )
    )
arg_value_edges(::GraphPropose, _, ::InputNode) = ()
arg_value_edge(parentnode::InputNode, _, comp_in_idx, proposer_name) =
    Input(parentnode.name) => CompIn(:proposers => proposer_name, comp_in_idx)
arg_value_edge(::ProposeNode, parentname, comp_in_idx, proposer_name) =
    CompOut(:proposers => parentname, :value) => CompIn(:proposers => proposer_name, comp_in_idx)

# edges to perform pairwise multiplication of all the tracked probs
function multiplier_edges(p::GraphPropose)
    firstname, rest = Iterators.peel(values(p.addr_to_name))
    secondname, rest = Iterators.peel(rest)
    edges = Pair{<:CompOut, <:CompIn}[
        CompOut(:proposers => firstname, :prob) => CompIn(:multipliers => 1, 1),
        CompOut(:proposers => secondname, :prob) => CompIn(:multipliers => 1, 2)
    ]

    for (i, name) in zip(2:(num_proposers(p) - 1), rest)
        append!(edges, [
            CompOut(:multipliers => i - 1, :out) => CompIn(:multipliers => i, 1),
            CompOut(:proposers => name, :prob) => CompIn(:multipliers => i, 2)
        ])
    end
    return edges
end

# input/output edges
io_edges(p::GraphPropose) = Iterators.flatten((
        (
            CompOut(:multipliers => num_proposers(p) - 1, :out) => Output(:prob),
            CompOut(:proposers => p.output_node_name, :value) => Output(:value)
        ),
        trace_output_edges(p)
    ))
trace_output_edges(p::GraphPropose) = (
        CompOut(:proposers => name, :trace) => Output(:trace => addr)
        for (addr, name) in p.addr_to_name    
    )

############################################
# Compilation Gen Static → Propose Circuit #
############################################

function propose_circuit(ir::Gen.StaticIR, arg_domain_sizes::NamedTuple)
    @assert isempty(ir.trainable_param_nodes)
    @assert length(arg_domain_sizes) == length(ir.arg_nodes)

    nodes = Dict{Symbol, ProposeGraphNode}()
    addr_to_name = Dict{Symbol, Symbol}()
    domain_sizes = Dict{Symbol, Int}()
    for node in ir.nodes
        handle_node!(nodes, node, domain_sizes, addr_to_name, arg_domain_sizes)
    end

    GraphPropose(
        arg_domain_sizes,
        ir.return_node.name,
        nodes,
        addr_to_name
    )
end

function handle_node!(nodes, node::Gen.ArgumentNode, domain_sizes, _, arg_domain_sizes)
    nodes[node.name] = InputNode(node.name)
    domain_sizes[node.name] = arg_domain_sizes[node.name]
end
function handle_node!(nodes, node::Gen.StaticIRNode, domain_sizes, addr_to_name, _)
    parent_names = [p.name for p in node.inputs]
    proposer = propose_circuit(node, parent_sizes(node, parent_names, domain_sizes))
    nodes[node.name] = ProposeNode(proposer, parent_names)
    domain_sizes[node.name] = output_domain_size(proposer)
    if has_trace(proposer)
        addr_to_name[node.addr] = node.name
    end
end

# RandomChoiceNode and JuliaNode have indexed parents, while
# GenerativeFunctionCallNodes have named parents
parent_sizes(::Union{Gen.RandomChoiceNode, Gen.JuliaNode}, parent_names, domain_sizes) =
    Tuple(domain_sizes[name] for name in parent_names)
parent_sizes(::Gen.GenerativeFunctionCallNode, parent_names, domain_sizes) =
    (;(name => domain_sizes[name] for name in parent_names)...)

# `propose_circuit` for `StaticIRNode`s
propose_circuit(gn::Gen.GenerativeFunctionCallNode, ads) =
    propose_circuit(gn.generative_function, ads)
propose_circuit(rcn::Gen.RandomChoiceNode, ads) = propose_circuit(rcn.dist, ads)
propose_circuit(jn::Gen.JuliaNode, ads) = propose_circuit(jn.fn, ads)

# `propose_circuit` for user-facing types
propose_circuit(g::Gen.StaticIRGenerativeFunction, arg_domain_sizes) =
    propose_circuit(Gen.get_ir(typeof(g)), arg_domain_sizes)
propose_circuit(g::Gen.Distribution, _) =
    DistributionPropose(get_cpt(g))
propose_circuit(f::Function, arg_domain_sizes) =
    DeterministicPropose(arg_domain_sizes, f)