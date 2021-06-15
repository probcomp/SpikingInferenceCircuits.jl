struct ApplyGenFn{Op} <: CompositeGenFn{Op}
    kernel_circuits::Vector{<:GenFn}
    input_domains::Tuple
    kernel_output_domains::Vector{<:Domain}
    op::Op
end
input_domains(a::ApplyGenFn) = a.input_domains
output_domain(a::ApplyGenFn) = IndexedProductDomain(a.kernel_output_domains)
has_traceable_value(a::ApplyGenFn) = any(has_traceable_value(gf) for gf in a.kernel_circuits)
traceable_value(a::ApplyGenFn) = IndexedValues(traceable_value(gf) for gf in a.kernel_circuits)

operation(a::ApplyGenFn) = a.op

sub_gen_fns(a::ApplyGenFn) = Tuple(a.kernel_circuits)

_possible_prob_names(a::ApplyGenFn) = (
    i for (i, gf) in enumerate(a.kernel_circuits)
    if has_score_output(gf)
)
score_outputter_names(a::ApplyGenFn{Propose}) = _possible_prob_names(a)
score_outputter_names(a::ApplyGenFn{Generate}) = (
    i for i in _possible_prob_names(a)
    if !isempty(operation(a).observed_addrs[i])
)

# arg values and return value mapping
arg_edges(a::ApplyGenFn) = (
    Input(:inputs => i => j) => CompIn(:sub_gen_fns => j, :inputs => input_name)
    for j=1:length(a.kernel_circuits)    
        for (i, input_name) in enumerate(arg_names(a.kernel_circuits[j]))
)
ret_edges(a::ApplyGenFn) = (
    CompOut(:sub_gen_fns => j, :value) => Output(:value => j) for j=1:length(a.kernel_circuits)
)

addr_to_name(a::ApplyGenFn) = Dict(i => i for i=1:length(a.kernel_circuits))

### From Apply combinator ###
function gen_fn_circuit(a::DiscreteIRTransforms.ApplyCombinator.Apply, arg_domains::Tuple, op::Op) where {Op <: GenFnOp}
    @assert all(dom isa IndexedProductDomain for dom in arg_domains)

    vector_sizes = Set(length(dom.subdomains) for dom in arg_domains)
    n_repetitions = only(vector_sizes) # all arg vectors should be the same size!
    
    subdomains = [[dom.subdomains[i] for dom in arg_domains] for i=1:n_repetitions]
    
    kernel_circuits = [
        gen_fn_circuit(kernel, Tuple(doms), apply_subop(i, op))
        for (i, (kernel, doms)) in enumerate(zip(a.kernels, subdomains))
    ]

    return ApplyGenFn(
        kernel_circuits,
        arg_domains,
        Base.map(output_domain, kernel_circuits),
        op
    )
end

apply_subop(_, ::Propose) = Propose()
apply_subop(i, op::Generate) = Generate(op.observed_addrs[i])
