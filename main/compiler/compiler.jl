#=
Compiler from a subset of Gen generative functions to circuits
which (roughly) implement GFI functions.

First, we will just implement `propose`.  We will implement this for:
- Sampling from finite-domain discrete distributions
- Deterministic functions with finite domain
- Static generative functions where all variables are discrete with finite domain

This currently works when all of the following criteria are met:
1. All distributions in a Gen model are `CPT`s.
2. All nodes' inputs are integers in a range 1...n, where `n` is the domain size.
=#

using Gen
using Circuits

abstract type Propose <: GenericComponent end
input_var_domain_sizes(::Propose)::Tuple{Vararg{Int}} = error("Not implemented.")
output_var_domain_size(::Propose)::Int = error("Not implemented.")
Circuits.inputs(p::Propose) = IndexedValues(FiniteDomainValue(n) for n in input_var_domain_sizes(p))
Circuits.outputs(p::Propose) = NamedValues(:value => FiniteDomainValue(output_var_domain_size(p)), :prob => PositiveReal())

struct DeterministicPropose <: Propose
    input_var_domain_sizes::Tuple
    output_var_domain_size::Int
    fn::Function
end
function DeterministicPropose(input_var_domain_sizes::Tuple, fn::Function)
    possible_outputs = Set()
    for vals in Iterators.product(input_var_domain_sizes)
        push!(possible_outputs, fn(vals...))
    end
    DeterministicPropose(input_var_domain_sizes, length(possible_outputs), fn)
end

input_var_domain_sizes(d::DeterministicPropose) = d.input_var_domain_sizes
output_var_domain_size(d::DeterministicPropose) = d.output_var_domain_size
Circuits.implement(d::DeterministicPropose, ::Target) =
    propose_wrapper(CPTSampleScore(deterministic_cpt(d), true), d)

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

struct DistributionPropose <: Propose
    cpt::CPT
end
input_var_domain_sizes(d::DistributionPropose) = input_ncategories(d.cpt)
output_var_domain_size(d::DistributionPropose) = ncategories(d.cpt)

Circuits.implement(d::DistributionPropose, ::Target) = propose_wrapper(CPTSampleScore(d.cpt, true), d)

propose_wrapper(cpt_ss, d) = CompositeComponent(
        inputs(d), outputs(cpt_ss),
        (cpt_sample_score=cpt_ss,),
        Iterators.flatten((
            (Input(i) => CompIn(:cpt_sample_score, :in_vals => i) for i=1:length(inputs(d))),
            (
                CompOut(:cpt_sample_score, :value) => Output(:value),
                CompOut(:cpt_sample_score, :prob) => Output(:prob)
            )
        )),
        d
    )

### Propose from a graph of other propose circuits! ###

struct ProposeGraphNode
    propose::Propose
    parents::Vector{Int}
end

# For now we assume only one output; later we should support having multiple outputs.
struct GraphPropose <: Propose
    input_var_domain_sizes::Tuple
    output_node_idx::Int
    idx_to_node::Vector{Union{ProposeGraphNode, Input}}
end
input_var_domain_sizes(g::GraphPropose) = g.input_var_domain_sizes

# TODO: what if we just route an input through to the output?
output_var_domain_size(g::GraphPropose) =
    output_var_domain_size(g.idx_to_node[g.output_node_idx].propose)

# Question: is it better to have one big multiplier which multiplies everything,
# or lots of pairwise multipliers?
# This will depend on the implementation; for this abstract version, perhaps we want to leave this
# as an option?
Circuits.implement(p::GraphPropose, ::Target) =
    CompositeComponent(
        inputs(p), outputs(p),
        (
            proposers=IndexedComponentGroup((v.propose for v in p.idx_to_node)),
            multipliers=IndexedComponentGroup(
                PositiveRealMultiplier(2) for _=1:(length(p.idx_to_node) - 1)
            )
        ),
        Iterators.flatten((
            Iterators.flatten((
                parent_value_edges(p, idx, node)
                for (idx, node) in enumerate(p.idx_to_node)
            )),
            Iterators.flatten((
                multiplier_edges(i)
                for i=1:(length(p.idx_to_node) - 1)
            )),
            (
                CompOut(:multipliers => length(p.idx_to_node) - 1, :out) => Output(:prob),
                CompOut(:proposers => p.output_node_idx, :value) => Output(:value)
            )
        )),
        p
    )
parent_value_edges(p::GraphPropose, idx, node::ProposeGraphNode) = (
        let parent = p.idx_to_node[parentidx]
            let input_nodename = parent isa Input ? parent : CompOut(:proposers => parentidx, :value)
                input_nodename => CompIn(:proposers => idx, parentidx)
            end
        end
        for (parentidx, domain_size) in zip(node.parents, input_var_domain_sizes(node.propose))
    )

multiplier_edges(i) =
    if i == 1
        (
            CompOut(:proposers => 1, :prob) => CompIn(:multipliers => 1, 1),
            CompOut(:proposers => 2, :prob) => CompIn(:multipliers => 1, 2)
        )
    else
        (
            CompOut(:multipliers => i - 1, :out) => CompIn(:multipliers => i, 1),
            CompOut(:proposers => i + 1, :out) => CompIn(:multipliers => i, 2)
        )
    end

####
# From Gen
####
function propose_circuit(ir::Gen.StaticIR, arg_domain_sizes::Tuple{Vararg{Int}})
    @assert isempty(ir.trainable_param_nodes)
    @assert length(arg_domain_sizes) == length(ir.arg_nodes)

    nodes = []
    idx_to_domain_size = []
    name_to_idx = Dict(node.name => i for (i, node) in enumerate(ir.nodes))
    for (i, node) in enumerate(ir.nodes)
        handle_node!(nodes, idx_to_domain_size, i, node, arg_domain_sizes, name_to_idx)
    end

    GraphPropose(
        arg_domain_sizes,
        name_to_idx[ir.return_node.name],
        nodes
    ) 
end
function handle_node!(nodes, idx_to_domain_size, idx, ::Gen.ArgumentNode, arg_domain_sizes, name_to_idx)
    push!(nodes, Input(idx))
    idx_to_domain_size[idx] = arg_domain_sizes[idx]
end
function handle_node!(nodes, idx_to_domain_size, idx, node::Gen.StaticIRNode, _, name_to_idx)
    parent_indices = [name_to_idx[n.name] for n in node.inputs]
    parent_sizes = Tuple(idx_to_domain_size[i] for i in parent_indices)
    proposer = propose_circuit(node, parent_sizes)
    push!(nodes, ProposeGraphNode(proposer, parent_indices))
    push!(idx_to_domain_size, output_var_domain_size(proposer))
end

propose_circuit(g::Gen.StaticIRGenerativeFunction, arg_domain_sizes) =
    propose_circuit(Gen.get_ir(typeof(g)), arg_domain_sizes)
propose_circuit(gn::Gen.GenerativeFunctionCallNode, ads) =
    propose_circuit(gn.generative_function, ads)

propose_circuit(g::Gen.Distribution, _) =
    DistributionPropose(get_cpt(g))
propose_circuit(rcn::Gen.RandomChoiceNode, ads) = propose_circuit(rcn.dist, ads)

propose_circuit(f::Function, arg_domain_sizes) =
    DeterministicPropose(arg_domain_sizes, f)
propose_circuit(jn::Gen.JuliaNode, ads) = propose_circuit(jn.fn, ads)