# TODO: more explicitely document
# what the input/output addresses are for each circuit

"""
    abstract type GenFnOp end

An operation a generative function circuit can perform.
"""
abstract type GenFnOp end

"""
    Propose <: GenFnOp
    Propose()

[Propose](https://www.gen.dev/dev/ref/gfi/#Gen.propose) operation:
sample a trace and output the `1/P` where `P` is the probability
of having sampled that trace.  (Note that this returns `1/P`, whereas
in Gen Propose returns `P`.)
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

### Methods which GenFn sub-types should implement ###

"""
    operation(::GenFn)

The operation implemented by this generative function circuit.
"""
operation(::GenFn) = error("Not implemented.")
operation(::GenFn{Propose}) = Propose()

"""
    input_domains(::GenFn)

A tuple of `Domain`s giving the domain for each input of the GenFn.
"""
input_domains(::GenFn) = error("Not implemented.")

"""
    output_domain(::GenFn)

The output `Domain` for the GenFn.
"""
output_domain(::GenFn) = error("Not implemented.")

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

"""
    score_value(::GenFn)

The `Value` which will be be output by this GenFn circuit _if_ there is a score output.
(If there is no score output, the output from this function does not matter [and
the function need not be implemented].)
Should have `NonnegativeReal` as an abstract version.

(E.g. this might be `SingleNonnegativeReal()` or a `ProductNonnegativeReal`.)
"""
score_value(::GenFn) = error("Not implemented.")

### Methods which are automatically implemented using the above ###

"""
    arg_names(g::GenFn)

Iterator over the names of the arguments to the generative function, in order.
(It will hold that `g` receives inputs at the names given by this function.)
"""
arg_names(g::GenFn) = keys(input_domains(g))

## trace & prob output ##
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
function trace_value(g::GenFn)
    @assert has_trace(g)
    return _trace_value(g)
end
_trace_value(g::GenFn{Propose}) = traceable_value(g)
_trace_value(g::GenFn{Generate}) =
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
    has_score_output(::GenFn)::Bool

Whether this gen fn circuit outputs a score (probability or reciprocal probability) (`:score`).
"""
has_score_output(g::GenFn{Propose}) = has_traceable_value(g)
# if there are traceable values whose probabilities we can access,
# and any of these traceable values are observed, generate will output a prob
# (if _all_ values are sampled and not observed, we don't output a score)
has_score_output(g::GenFn{Generate}) = has_traceable_value(g) && !isempty(operation(g).observed_addrs)

## GenFn input/output ##
_input_val(g) = CompositeValue(map(to_value, input_domains(g)))
Circuits.inputs(g::GenFn{Propose}) = NamedValues(
    :inputs => _input_val(g)
)
Circuits.inputs(g::GenFn{Generate}) = NamedValues(
    :inputs => _input_val(g),
    (isempty(operation(g).observed_addrs) ? () : 
        (:obs => get_selected(traceable_value(g), operation(g).observed_addrs),)
    )...
)

Circuits.outputs(g::GenFn) = NamedValues(
    :value => to_value(output_domain(g)),
    (has_trace(g) ? (:trace => trace_value(g),) : ())...,
    (has_score_output(g) ? (:score => score_value(g),) : ())...
)

### Construct GenFn circuit from a Gen/Julia object ###

"""
    gen_fn_circuit(object_to_convert_to_circuit, arg_domains::Tuple{Vararg{<:Domain}}, op::Op)
    gen_fn_circuit(object_to_convert_to_circuit, arg_domains::NamedTuple{Vararg{<:Domain}}, op::Op)

Get a GenFn component for the given `object_to_convert_to_circuit` running the operation `op`.
The circuit will expect input values to be in the given `arg_domain`s.

(For static generative functions, `arg_domains` should be a NamedTuple using the argument names;
for other objects, `arg_domains` should just be a tuple.)
"""
gen_fn_circuit(_, _, _) = error("Not implemented")

"""
    gf::ImplementableGenFn

A generative function packaged with enough metadata that it can be implemented as a stochastic
digital circuit.
To implement it to perform operation `op`, call `gen_fn_circuit(gf, op::GenFnOp)`.
"""
abstract type ImplementableGenFn end
icpts(::ImplementableGenFn) = error("Not implemented.")
input_domains(::ImplementableGenFn) = error("Not implemented.")

"""
    gen_fn_circuit(gf::ImplementableGenFn, op::GenFnOp)

A GenFnCircuit for the given generative function with associated metadata (including input domain labels).
"""
gen_fn_circuit(gf::ImplementableGenFn, op::GenFnOp) = 
    gen_fn_circuit(
        gf |> icpts |> DiscreteIRTransforms.inline_constant_nodes,
        input_domains(gf),
        op
    )

"""
    GenFnWithInputDomains(gen_fn::GenerativeFunction, input_domains::Vector{DiscreteIRTransforms.Domain})

A generative function with labels giving the domain of each input value to the gen fn.
"""
struct GenFnWithInputDomains <: ImplementableGenFn
    gen_fn::GenerativeFunction
    input_domains::Vector{<:DiscreteIRTransforms.Domain}
end
GenFnWithInputDomains(gen_fn::GenerativeFunction, input_domains) =
    GenFnWithInputDomains(gen_fn,
        [
            if dom isa DiscreteIRTransforms.Domain
                dom
            else
                DiscreteIRTransforms.EnumeratedDomain(dom)
            end
            for dom in input_domains
        ]
    )

icpts(gf::GenFnWithInputDomains) = to_indexed_cpts(gf.gen_fn, gf.input_domains)[1]
input_domains(gf::GenFnWithInputDomains) = [to_gf_indexed_domain(dom) for dom in gf.input_domains]

replace_return_node(gf::GenFnWithInputDomains) =
    GenFnWithInputDomains(DiscreteIRTransforms.replace_return_node(gf.gen_fn), gf.input_domains)

"""
    WithActivatorInput(gf::GenFnWithInputDomains, activator_input_name)

An `ImplementableGenFn` equivalent to `gf`, except that before being implemented,
a argument with domain `[1]` is added to the generative function with name `activator_input_name`,
and the argument is fed as input into every node which currently has no arguments.
"""
struct WithActivatorInput <: ImplementableGenFn
    gf::GenFnWithInputDomains
    activator_input_name
end
add_activator_input(gf::GenFnWithInputDomains, activator_input_name) =
    WithActivatorInput(gf, activator_input_name)

icpts(gf::WithActivatorInput) = DiscreteIRTransforms.add_activator_input(icpts(gf.gf), gf.activator_input_name)
input_domains(gf::WithActivatorInput) = [
    FiniteDomain(1), # activator input
    [to_gf_indexed_domain(dom) for dom in gf.gf.input_domains]...
]

replace_return_node(gf::WithActivatorInput) = WithActivatorInput(replace_return_node(gf.gf), gf.activator_input_name)

struct TruncatedIGF <: ImplementableGenFn
    minprob::Float64
    gf::ImplementableGenFn
end
icpts(gf::TruncatedIGF)         = DiscreteIRTransforms.truncate(icpts(gf.gf), gf.minprob)
input_domains(gf::TruncatedIGF) = input_domains(gf.gf)
truncate_implementable_gf(gf::ImplementableGenFn, minprob) = TruncatedIGF(minprob, gf)