struct MapGenFn{Op} <: GenFn{Op}
    kernel_circuits::Vector{GenFn}
    input_domains::Tuple
    kernel_output_domains::Vector{Domain}
    op::Op
end
input_domains(m::MapGenFn) = m.input_domains
output_domain(m::MapGenFn) = IndexedProductDomain(m.kernel_output_domains)
has_traceable_value(m::MapGenFn) = any(has_traceable_value(gf) for gf in m.kernel_circuits)
traceable_value(m::MapGenFn) = IndexedValues(i => traceable_value(gf) for (i, gf) in enumerate(m.kernel_circuits))
operation(m::MapGenFn) = m.op

### Implementation ###
Circuits.implement(m::MapGenFn, ::Target) = CompositeComponent(
    inputs(m), outputs(m),
    (
        sub_gen_fns=IndexedComponentGroup(m.kernel_circuits),
        (let multgroup = multipliers_group(length(m.kernels)) # `multipliers_group` defined in `graph.jl`
            isempty(multgroup) ? () : (:multipliers => multgroup,)
        end)...
    ),
    Iterators.flatten((
        multiplier_edges(g), # from `graph.jl`
        map_io_edges(g)
    )),
    g
)

# TODO: is there a way to refactor to merge some of this code with code in `graph.jl`?
_possible_prob_names(m::MapGenFn) = (
    i for (i, gf) in enumerate(m.kernel_circuits)
    if has_prob_output(gf)
)
prob_outputter_names(m::MapGenFn{Propose}) = _possible_prob_names(m)
prob_outputter_names(m::MapGenFn{Generate}) = (
    i for i in _possible_prob_indices(m)
    if !isempty(operation(g).observed_addrs[i])
)
num_internal_prob_outputs(m::MapGenFn) = length(collect(prob_outputter_names(m)))

# TODO
map_io_edges(m::MapGenFn) = Iterators.flatten((
    # probability output
    # trace output
    # 
))

# arg values and return value mapping
arg_ret_edges(m::MapGenFn) = Iterators.flatten((arg_edges(m), ret_edges(m)))
arg_edges(m::MapGenFn) = (
    Input(i => j) => CompIn(j, i) for i=1:length(input_domains(m)) for j=1:length(m.kernel_circuits)
)
ret_edges(m::MapGenFn) = (
    CompOut(j, :value) => Output(:value => j) for j=1:length(m.kernel_circuits)
)

### From Map combinator ###
function gen_fn_circuit(map::Gen.Map, arg_domains::Tuple, op::Op) where {Op <: GenFnOp}
    @assert all(dom isa IndexedProductDomain for dom in arg_domains)

    vector_sizes = Set(length(dom.subdomains) for dom in arg_domains)
    n_repetitions = only(vector_sizes) # all arg vectors should be the same size!
    
    subdomains = [[dom.subdomains[i] for dom in arg_domains] for i=1:n_repetitions]
    
    kernel_circuits = [
        gen_fn_circuit(map.kernel, doms, map_subop(i, op))
        for (i, doms) in enumerate(subdomains)
    ]

    return MapGenFn(
        kernel_circuits,
        arg_domains,
        map(output_domain, kernel_circuits),
        op
    )
end

map_subop(_, ::Propose) = Propose()
map_subop(i, op::Generate) = Generate(op.observed_addrs[i])
