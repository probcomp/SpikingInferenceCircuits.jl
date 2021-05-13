struct LabeledDomainGF
    gf::Gen.GenerativeFunction
    labels::Vector{DiscreteIRTransforms.EnumeratedDomain}
end
LabeledDomainGF(gf::Gen.GenerativeFunction, labels::Vector{<:Vector}) =
    LabeledDomainGF(gf, map(EnumeratedDomain, labels))

function gen_fn_circuit_and_bijs(lgf::LabeledDomainGF, op::GenFnOp)
    cpts, bijs = to_indexed_cpts(lgf.gf, lgf.labels)
    circuit = gen_fn_circuit(cpts, map(FiniteDomain ∘ length ∘ DiscreteIRTransforms.vals, lgf.labels), op)
    return (circuit, bijs)
end
gen_fn_circuit(lgf::LabeledDomainGF, op::GenFnOp) = first(gen_fn_circuit_and_bijs(lgf, op))