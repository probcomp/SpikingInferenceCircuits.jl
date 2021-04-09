"""
    CompositeGenFn{Op} <: GenFn{Op}

A GenFn circuit built out of other gen fn circuits.

To use the `CompositeGenFn` `Circuits.implement`, a subtype must implement
`sub_gen_fns`, `prob_outputter_names`, `arg_edges`, `ret_edges`, and `addr_to_name`,
in addition to the required methods for a `GenFn`.
"""
abstract type CompositeGenFn{Op} <: GenFn{Op} end

"""
    sub_gen_fns(::CompositeGenFn)

Tuple or NamedTuple of subsidiary generative functions.
"""
sub_gen_fns(::CompositeGenFn) = error("Not implemented.")

"""
    prob_outputter_names(g::CompositeGenFn)

Iterator over indices or names of the sub gen fns in `sub_gen_fns(g)`
which output a probability.
"""
prob_outputter_names(::CompositeGenFn) = error("Not implemented.")

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

num_internal_prob_outputs(g) = length(collect(prob_outputter_names(g)))

Circuits.implement(g::CompositeGenFn, ::Target) =
    CompositeComponent(
        inputs(g), outputs(g),
        (
            sub_gen_fns=ComponentGroup(sub_gen_fns(g)),
            (let multgroup = multipliers_group(g)
                isempty(multgroup.subcomponents) ? () : (:multipliers => multgroup,)
            end)...
        ),
        Iterators.flatten((
            arg_edges(g),
            ret_edges(g),
            multiplier_edges(g),
            io_edges(g)
        )),
        g
    )

# TODO: explore design tradeoffs between using pairwise multiplication vs multiple-input multiplication
multipliers_group(g) = IndexedComponentGroup(
    PositiveRealMultiplier(2) for _=1:(num_internal_prob_outputs(g) - 1)
)

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
io_edges(g::CompositeGenFn) = Iterators.flatten((
        (   # probability output
            if has_prob_output(g)
                if num_internal_prob_outputs(g) > 1
                    (CompOut(:multipliers => num_internal_prob_outputs(g) - 1, :out) => Output(:prob),)
                else
                    (CompOut(:sub_gen_fns => first(prob_outputter_names(g)), :prob) => Output(:prob),)
                end
            else
                ()
            end
        ),
        trace_output_edges(g),
        obs_input_edges(g)
    ))
trace_output_edges(g::CompositeGenFn) = (
        CompOut(:sub_gen_fns => addr_to_name(g)[addr], :trace) => Output(:trace => addr)
        for addr in keys(trace_value(g))
    )
obs_input_edges(::CompositeGenFn{Propose}) = ()
obs_input_edges(g::CompositeGenFn{Generate}) = (
    Input(:obs => addr) => CompIn(:sub_gen_fns => addr_to_name(g)[addr], :obs)
    for (addr, _) in addr_to_name(g) if !isempty(operation(g).observed_addrs[addr])
)