### get_selected util ###

"""
    get_selected(value::Value, sel::Selection)

Filters a nested value to only include selected nodes.
Returns a value which filters the given `value` 
so that the only leaf nodes are those at addresses selected in `sel`.
"""
get_selected(val::FiniteDomainValue, ::AllSelection) = val
get_selected(::FiniteDomainValue, ::EmptySelection) = error()
function get_selected(val::FiniteDomainValue, s::ComplementSelection)
    @assert s.complement == EmptySelection() "complement was $(s.complement)"
    val
end
get_selected(val::CompositeValue, sel::Selection) = NamedValues((
        addr => get_selected(subval, sel[addr])
        for (addr, subval) in pairs(val)
        if has_selected(subval, sel[addr])
    )...)

function has_selected(::FiniteDomainValue, s::ComplementSelection)
    if isempty(s.complement)
        return true
    elseif s.complement == AllSelection()
        return false
    else
        error("unexpected")
    end
end
has_selected(_, ::AllSelection) = true
has_selected(_, ::EmptySelection) = false
has_selected(v::CompositeValue, s::Selection) = any(
        has_selected(sv, s[a]) for (a, sv) in pairs(v)
    )

### types ###

"""
    abstract type GenFnOp end

An operation a generative function circuit can perform.
"""
abstract type GenFnOp end
"""
    Propose <: GenFnOp
    Propose()

[Propose](https://www.gen.dev/dev/ref/gfi/#Gen.propose) operation:
sample a trace and output the probability
of having sampled that trace.
(No probability is output for a deterministic generative function.)
Also outputs the return value of this generative function execution.
"""
struct Propose <: GenFnOp end
"""
    Generate <: GenFnOp
    Generate(observed_addrs::Set)

[Generate](https://www.gen.dev/dev/ref/gfi/#Gen.generate) operation:
sample all values in a trace
except those which are observed, whose addresses are given by
`observed_addrs`.  Output the probability of the gen fn
producing the observed values, given the newly-generated values.
Also outputs the return value of this generative function execution.
"""
struct Generate <: GenFnOp
    observed_addrs::Selection
end

"""
    Assess()

[Assess](https://www.gen.dev/dev/ref/gfi/#Gen.assess) operation:
given the value for each traceable choice, outputs the probability of sampling
these values.
Also outputs the return value of this generative function execution.
(This is equivalent to, and implemented as, `Generate(AllSelection())`.)
"""
Assess() = Generate(AllSelection())

"""
    abstract type GenFn{Op} <: GenericComponent end

Circuit which implements an operation of type `Op` for a generative function.
The specific operation is given by `operation(::GenFn)`.
"""
abstract type GenFn{Op} <: GenericComponent end
"""
    operation(::GenFn)

The operation implemented by this generative function circuit.
"""
operation(::GenFn) = error("Not implemented.")
operation(::GenFn{Propose}) = Propose()
"""
    input_domain_sizes(::GenFn)

A `Tuple` or `NamedTuple` giving the sizes of the domains
for each input value to this generative function.
"""
input_domain_sizes(::GenFn) = error("Not implemented.")
"""
    output_domain_size(::GenFn)::Int

The size of the output value of this generative function.
"""
output_domain_size(::GenFn)::Int = error("Not implemented.")

"""
    has_traceable_value(::GenFn)::Bool

Whether this generative function has any random choices which could be
output in the trace.  This is `true` even if the traceable values are not traced
in the operation this `GenFn` circuit implements (e.g. if the values are observed in
this operation).
"""
has_traceable_value(::GenFn) = error("Not implemented.")

"""
    traceable_value(::GenFn)

A `Value` containing all the traceable values which can be produced
by this Generative function.  Should be a `FiniteDomainValue`
for a top-level value, or a `CompositeValue` with sub-values at addresses.
"""
traceable_value(::GenFn) = error("Not implemented.")

### trace & prob output ###
"""
    has_trace(::GenFn)::Bool

Whether this generative function operation outputs a trace.
"""
has_trace(g::GenFn{Propose}) = has_traceable_value(g)
has_trace(g::GenFn{Generate}) = (
    has_traceable_value(g) && has_selected(traceable_value(g), Gen.complement(operation(g).observed_addrs))
)
"""
    trace_value(g::GenFn)

The `Value` at the `:trace` output for this circuit.  This should only be called
if `has_trace(g)`.
"""
trace_value(g::GenFn{Propose}) = traceable_value(g)
trace_value(g::GenFn{Generate}) =
    let tv = traceable_value(g)
        if tv isa FiniteDomainValue
            @assert isempty(operation(g).observed_addrs)
            tv
        else
            @assert tv isa CompositeValue
            get_selected(tv, Gen.complement(operation(g).observed_addrs))
        end
    end

"""
    has_prob_output(::GenFn)::Bool

Whether this gen fn circuit outputs a probability (`:prob`).
"""
has_prob_output(::GenFn{Propose}) = true
# if there are traceable values whose probabilities we can access,
# and any of these traceable values are observed, generate will output a prob
# (if _all_ values are sampled and not observed, we don't output a score)
has_prob_output(g::GenFn{Generate}) = has_traceable_value(g) && !isempty(operation(g).observed_addrs)

### GenFn input/output ###
_inputvals(g::GenFn) = CompositeValue(
    map(FiniteDomainValue, input_domain_sizes(g))
)
Circuits.inputs(g::GenFn{Propose}) = NamedValues(
    :inputs => _inputvals(g)
)
Circuits.inputs(g::GenFn{Generate}) = NamedValues(
    :inputs => _inputvals(g),
    (isempty(operation(g).observed_addrs) ? () : 
        (:obs => get_selected(traceable_value(g), operation(g).observed_addrs),)
    )...
)

Circuits.outputs(g::GenFn) = NamedValues(
    :value => FiniteDomainValue(output_domain_size(g)),
    (has_trace(g) ? (:trace => trace_value(g),) : ())...,
    (has_prob_output(g) ? (:prob => PositiveReal(),) : ())...
)

### gen_fn leaf node implementor ###
"""
    genfn_from_cpt_sample_score(cpt_ss::CPTSampleScore, g::GenFn, val_to_trace::Bool)

A `CompositeComponent` which implements `pg` using the given `cpt_ss`.  If `val_to_trace` is true,
the value output from `cpt_ss` is both output as `:value` and `:trace`; if this is false, it is only
output as `:value`.
"""
genfn_from_cpt_sample_score(cpt_ss, g, val_to_trace) = CompositeComponent(
        inputs(g), outputs(g),
        (cpt_sample_score=cpt_ss,),
        Iterators.flatten((
            (Input(:inputs => i) => CompIn(:cpt_sample_score, :in_vals => i) for i=1:length(inputs(g)[:inputs])),
            (
                CompOut(:cpt_sample_score, :value) => Output(:value),
                (has_prob_output(g) ? (CompOut(:cpt_sample_score, :prob) => Output(:prob),) : ())...,
                (val_to_trace ? (CompOut(:cpt_sample_score, :value) => Output(:trace),) : ())...
            )
        )),
        g
    )

#################
# Deterministic #
#################
"""
    DeterministicGenFn{Op} <: GenFn{Op}
    DreterministicDomainFn{Op}(input_domain_sizes::Tuple, fn::Function)

Circuit to perform an operation for a deterministic generative function (given by function `fn`, where the arguments
have the given input domain sizes).
"""
struct DeterministicGenFn{Op} <: GenFn{Op}
    input_domain_sizes::Tuple
    output_domain_size::Int
    fn::Function
end
function DeterministicGenFn{Op}(input_domain_sizes::Tuple, fn::Function) where {Op <: GenFnOp}
    output_domain_size = length(unique(fn(vals...) for vals in Iterators.product(input_var_domain_sizes)))
    DeterministicGenFn{Op}(input_domain_sizes, output_domain_size, fn)
end
input_domain_sizes(g::DeterministicGenFn) = g.input_domain_sizes
output_domain_size(g::DeterministicGenFn) = g.output_domain_size
has_traceable_value(::DeterministicGenFn) = false
# currently, I assume we never observe a deterministic node (since these are not traced)
operation(::DeterministicGenFn{Generate}) = Generate(Set())

### implementation ###

# CPT with deterministic outputs
function deterministic_cpt(d::DeterministicGenFn)
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

Circuits.implement(g::DeterministicGenFn, ::Target) =
    genfn_from_cpt_sample_score(
        CPTSampleScore(deterministic_cpt(g), true),
        g, false
    )

################
# Distribution #
################

struct DistributionGenFn{Op} <: GenFn{Op}
    is_observed::Bool
    cpt::CPT
    DistributionGenFn{Propose}(cpt::CPT) = new{Propose}(false, cpt)
    DistributionGenFn{Generate}(is_observed::Bool, cpt::CPT) = new{Generate}(is_observed, cpt)
end
DistributionGenFn(cpt::CPT, op::Propose) = DistributionGenFn{Propose}(cpt)
DistributionGenFn(cpt::CPT, op::Generate) = DistributionGenFn{Generate}(op.observed_addrs == AllSelection(), cpt)

input_domain_sizes(d::DistributionGenFn) = input_ncategories(d.cpt)
output_domain_size(d::DistributionGenFn) = ncategories(d.cpt)
has_traceable_value(d::DistributionGenFn) = true
traceable_value(d::DistributionGenFn) = FiniteDomainValue(output_domain_size(d))
operation(d::DistributionGenFn{Generate}) = Generate(d.is_observed ? AllSelection() : EmptySelection())
Circuits.implement(d::DistributionGenFn, ::Target) =
    genfn_from_cpt_sample_score(CPTSampleScore(d.cpt, true), d, !d.is_observed)

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
output_domain_size(g::GenFnNode, _) = output_domain_size(g.gen_fn)
output_domain_size(i::InputNode, g) = input_domain_size(g)[i.name]

# TODO: docstring
struct GraphGenFn{Op} <: GenFn{Op}
    input_domain_sizes::NamedTuple
    output_node_name::Symbol
    nodes::Dict{Symbol, GenFnGraphNode}
    addr_to_name::Dict{Symbol, Symbol}
    observed_addrs::Selection
    GraphGenFn{Propose}(ids, ons, n, a) = new{Propose}(ids, ons, n, a, EmptySelection())
    GraphGenFn{Generate}(ids, ons, n, a, oa::Selection) = new{Generate}(ids, ons, n, a, oa)
end
GraphGenFn(ids, ons, n, a, ::Propose) = GraphGenFn{Propose}(ids, ons, n, a)
GraphGenFn(ids, ons, n, a, op::Generate) = GraphGenFn{Generate}(ids, ons, n, a, op.observed_addrs)

input_domain_sizes(g::GraphGenFn) = g.input_domain_sizes
output_domain_size(g::GraphGenFn) = output_domain_size(g.nodes[g.output_node_name], g)
has_traceable_value(g::GraphGenFn) = any(has_traceable_value(g.nodes[name].gen_fn) for name in values(g.addr_to_name))
traceable_value(g::GraphGenFn) = NamedValues((
        addr => traceable_value(g.nodes[name].gen_fn)
        for (addr, name) in g.addr_to_name
    )...)
operation(g::GraphGenFn{Generate}) = Generate(g.observed_addrs)

# during `propose`, we score every traceable sub-gen-fn;
# during `generate`, we only score those traceable sub-gen-fns which we observe
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
multipliers_group(g::GraphGenFn) = IndexedComponentGroup(
        PositiveRealMultiplier(2) for _=1:(num_internal_prob_outputs(g) - 1)
    )

# edges from argument values --> proposer input
arg_value_edges(g::GraphGenFn) = Iterators.flatten(
    arg_value_edges(g, name, node) for (name, node) in g.nodes
)
arg_value_edges(g::GraphGenFn, name, node::GenFnNode) = (
    arg_value_edge(g.nodes[parentname], parentname, comp_in_idx, name)
    for (comp_in_idx, (parentname, domain_size)) in enumerate(
        zip(node.parents, input_domain_sizes(node.gen_fn))
    )
)
arg_value_edges(::GraphGenFn, _, ::InputNode) = ()
arg_value_edge(parentnode::InputNode, _, comp_in_idx, gen_fn_name) =
    Input(:inputs => parentnode.name) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)
arg_value_edge(::GenFnNode, parentname, comp_in_idx, gen_fn_name) =
    CompOut(:sub_gen_fns => parentname, :value) => CompIn(:sub_gen_fns => gen_fn_name, :inputs => comp_in_idx)

# edges to perform pairwise multiplication of all the tracked probs
function multiplier_edges(g::GraphGenFn)
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

##########################################
# Compilation Gen Static → GenFn Circuit #
##########################################

function gen_fn_circuit(ir::Gen.StaticIR, arg_domain_sizes::NamedTuple, op::Op) where {Op <: GenFnOp}
    @assert isempty(ir.trainable_param_nodes)
    @assert length(arg_domain_sizes) == length(ir.arg_nodes)

    nodes = Dict{Symbol, GenFnGraphNode{Op}}()
    addr_to_name = Dict{Symbol, Symbol}()
    domain_sizes = Dict{Symbol, Int}()
    for node in ir.nodes
        handle_node!(nodes, node, domain_sizes, addr_to_name, arg_domain_sizes, op)
    end

    GraphGenFn(
        arg_domain_sizes,
        ir.return_node.name,
        nodes,
        addr_to_name,
        op
    )
end

function handle_node!(nodes, node::Gen.ArgumentNode, domain_sizes, _, arg_domain_sizes, op::Op) where {Op <: GenFnOp}
    nodes[node.name] = InputNode{Op}(node.name)
    domain_sizes[node.name] = arg_domain_sizes[node.name]
end
function handle_node!(nodes, node::Gen.StaticIRNode, domain_sizes, addr_to_name, _, op::Op) where {Op <: GenFnOp}
    parent_names = [p.name for p in node.inputs]
    sub_gen_fn = gen_fn_circuit(node, parent_sizes(node, parent_names, domain_sizes), subop(node, op))
    nodes[node.name] = GenFnNode(sub_gen_fn, parent_names)
    domain_sizes[node.name] = output_domain_size(sub_gen_fn)
    if has_traceable_value(sub_gen_fn)
        addr_to_name[node.addr] = node.name
    end
end

# figure out the operation for a sub-generative-function, given the op for the top-level gen fn
subop(_, ::Propose) = Propose()
subop(::Gen.JuliaNode, ::Generate) = Generate(EmptySelection())
subop(n::Gen.RandomChoiceNode, op::Generate) = Generate(n.addr in op.observed_addrs ? AllSelection() : EmptySelection())
subop(n::Gen.GenerativeFunctionCallNode, op::Generate) = Generate(op.observed_addrs[n.addr])

# RandomChoiceNode and JuliaNode have indexed parents, while
# GenerativeFunctionCallNodes have named parents
parent_sizes(::Union{Gen.RandomChoiceNode, Gen.JuliaNode}, parent_names, domain_sizes) =
    Tuple(domain_sizes[name] for name in parent_names)
parent_sizes(::Gen.GenerativeFunctionCallNode, parent_names, domain_sizes) =
    (;(name => domain_sizes[name] for name in parent_names)...)

# `gen_fn_circuit` for `StaticIRNode`s
gen_fn_circuit(gn::Gen.GenerativeFunctionCallNode, ads, op) =
    gen_fn_circuit(gn.generative_function, ads, op)
gen_fn_circuit(rcn::Gen.RandomChoiceNode, ads, op) = gen_fn_circuit(rcn.dist, ads, op)
gen_fn_circuit(jn::Gen.JuliaNode, ads, op) = gen_fn_circuit(jn.fn, ads, op)

# `gen_fn_circuit` for user-facing types
gen_fn_circuit(g::Gen.StaticIRGenerativeFunction, arg_domain_sizes, op) =
    gen_fn_circuit(Gen.get_ir(typeof(g)), arg_domain_sizes, op)
gen_fn_circuit(g::CPT, _, op) =
    DistributionGenFn(g, op)
gen_fn_circuit(::Gen.Distribution, _, _) = error("To be compiled to a circuit, all distributions must be CPTs.")
gen_fn_circuit(f::Function, arg_domain_sizes, ::Op) where {Op <: GenFnOp} =
    DeterministicGenFn{Op}(arg_domain_sizes, f)

#=
gen_fn_circuit(m::Gen.Map, arg_domain_sizes, op)

=#