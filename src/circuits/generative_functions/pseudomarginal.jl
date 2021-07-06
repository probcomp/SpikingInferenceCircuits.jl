# Currently only supports Assess mode
struct PseudoMarginalizedGenFn <: GenFn{Generate}
    model       :: GenFnCircuit
    proposal    :: GenFnCircuit
    n_particles :: Int
end
operation(::PseudoMarginalizedGenFn) = Assess()
input_domains(d::PseudoMarginalizedGenFn) = input_domains(d.model)
output_domain(d::PseudoMarginalizedGenFn)::Int = output_domains(d.model)
has_traceable_value(::PseudoMarginalizedGenFn) = true
traceable_value(::PseudoMarginalizedGenFn) = error("TODO")
score_value(::PseudoMarginalizedGenFn) = error("TODO")

gen_fn_circuit(::PseudoMarginalizedDist, arg_domains, op::Propose) = error("Pseudo-Marginalized dists in `Propose` mode are currently not supported.")
gen_fn_circuit(::PseudoMarginalizedDist, arg_domains, op::Generate) =
    if op == Assess()
        PseudoMarginalizedGenFn(
            gen_fn_circuit(d.model, arg_domains, Assess()),
            gen_fn_circuit(d.propose, arg_domains, Propose()),
            d.n_particles
        )
    else
        error("Pseudo-Marginalized dists are currently only supported in `Assess` mode (but gen_fn_circuit called with op = $op).")
    end
Circuits.implement(::PseudoMarginalizedGenFn, ::Target) = error("TODO")