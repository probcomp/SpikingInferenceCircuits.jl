"""
    CompositeGenFn{Op} <: GenFn{Op}

A GenFn circuit built out of other gen fn circuits.

To use the `CompositeGenFn` `Circuits.implement`, a subtype must implement
`sub_gen_fns`, `score_outputter_names`, `arg_edges`, `ret_edges`, and `addr_to_name`,
in addition to the required methods for a `GenFn` which are not automatically implemented
for a CompositeGenFn.

The automatically implemented methods for a CompositeGenFn are
- `Circuits.implement`
- `score_value`
"""
abstract type CompositeGenFn{Op} <: GenFn{Op} end

"""
    sub_gen_fns(::CompositeGenFn)

Tuple or NamedTuple of subsidiary generative functions.
"""
sub_gen_fns(::CompositeGenFn) = error("Not implemented.")

"""
    score_outputter_names(g::CompositeGenFn)

Iterator over indices or names of the sub gen fns in `sub_gen_fns(g)`
which output a probability.
"""
score_outputter_names(::CompositeGenFn) = error("Not implemented.")

"""
    arg_edges(::CompositeGenFn)

Iterator over edges to map inputs into sub gen fn nodes.

The implementation may rely on the sub gen fn with name/idx `n`
being at `:sub_gen_fns => n`.  (So the edges
may reference `CompIn(:sub_gen_fns => n, ...)` or
`CompOut(:sub_gen_fns => n, ...)`.)
"""
arg_edges(::CompositeGenFn) = error("Not implemented.")

"""
    ret_edges(::CompositeGenFn)

Iterator over edges needed to output the value from this gen fn.

The implementation may rely on the sub gen fn with name/idx `n`
being at `:sub_gen_fns => n`.  (So the edges
may reference `CompIn(:sub_gen_fns => n, ...)` or
`CompOut(:sub_gen_fns => n, ...)`.)
"""
ret_edges(::CompositeGenFn) = error("Not implemented")

"""
    addr_to_name(g::CompositeGenFn)

A `Dict` mapping from the trace address of any traceable sub gen fn
to the name or index in `sub_gen_fns(g)` of the gen fn with that address.
"""
addr_to_name(::CompositeGenFn) = error("Not implemented")

num_internal_score_outputs(g) = length(collect(score_outputter_names(g)))

function Circuits.implement(g::CompositeGenFn, ::Target)
    println("Implementing genfn of type $(typeof(g))...")
    component = CompositeComponent(
        inputs(g),
        # implement the `score` output so we can output factors
        # to the sub-score for each factors
        implement(outputs(g), Spiking(), :score),
        (
            sub_gen_fns=ComponentGroup(sub_gen_fns(g)),
        ),
        Iterators.flatten((
            # implemented by sub-type of CompositeGenFn:
            arg_edges(g),
            ret_edges(g),
            # implemented for all CompositeGenFn (using other methods defined for the sub-types):
            obs_input_edges(g),
            score_output_edges(g),
            trace_output_edges(g)
        )),
        g
    )
    println("Done implementing genfn of type $(typeof(g)).")
    return component
end

obs_input_edges(::CompositeGenFn{Propose}) = ()
obs_input_edges(g::CompositeGenFn{Generate}) = (
    Input(:obs => addr) => CompIn(:sub_gen_fns => addr_to_name(g)[addr], :obs)
    for (addr, _) in addr_to_name(g) if !isempty(operation(g).observed_addrs[addr])
)

trace_output_edges(g::CompositeGenFn) = 
    if has_trace(g)
        (
            CompOut(:sub_gen_fns => addr_to_name(g)[addr], :trace) => Output(:trace => addr)
            for addr in keys(trace_value(g))
        )
    else
        ()
    end

score_output_edges(g::CompositeGenFn) = (
        CompOut(:sub_gen_fns => name, :score) => Output(:score => name)
        for name in score_outputter_names(g)
    )

# Part of `GenFn` interface (not used for `implement`):
score_value(g::CompositeGenFn) = ProductNonnegativeReal(
        tuple_or_namedtuple(
            name => score_value(sub_gen_fns(g)[name])
            for name in score_outputter_names(g)
        )
    )

# internal utility function:
"""
Given an iterator over `key => value` pairs, if all keys are integers,
returns a Tuple of the values, and if all keys are Symbols, returns
a NamedTuple from the keys to the values.
(Errors if the keys are not either all Ints or all Symbols.)
"""
tuple_or_namedtuple(itr) =
    if all(name_val[1] isa Int for name_val in itr)
        (Iterators.map(name_val -> name_val[2], itr)...,)
    else
        @assert all(name_val[1] isa Symbol for name_val in itr)
        (;itr...)
    end
