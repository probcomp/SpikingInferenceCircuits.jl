# Currently only supports Assess mode, and currently only supports 1 particle
struct PseudoMarginalizedGenFn <: GenFn{Generate}
    particle          
    in_domains
    out_domain
    output_var_addr
end
operation(::PseudoMarginalizedGenFn)           = Assess()
input_domains(d::PseudoMarginalizedGenFn)      = d.in_domains
output_domain(d::PseudoMarginalizedGenFn)      = d.out_domain
has_traceable_value(::PseudoMarginalizedGenFn) = true
traceable_value(d::PseudoMarginalizedGenFn)    = to_value(d.out_domain) # we can trace directly at this gen fn's return
score_value(d::PseudoMarginalizedGenFn)        = outputs(d.particle)[:weight]

function pmdist_to_gen_fn_circuit(d, arg_domains, op::Generate)
    if op != Assess()
        error("Pseudo-Marginalized dists are currently only supported in `Assess` mode (but gen_fn_circuit called with op = $op).")
    elseif d.n_particles != 1
        error("Pseudo-Marginalized distributions are currently only compilable if they use exactly 1 particle (but this PM-Dist uses $(d.n_particles) particles).")
    end
    @assert (d.args_for_compilation isa Tuple && length(d.args_for_compilation) == 4) "Need 4 args to be able to compile PM-Dist: latent_domains, latent_addr_order, obs_addr_order, output_var_addr"
    (latent_domains, latent_addr_order, obs_addr_order, output_var_addr) = d.args_for_compilation

    proposal_circ = gen_fn_circuit(d.proposal, (arg_domains..., latent_domains...), Propose())
    latent_circ   = gen_fn_circuit(d.latent_model, arg_domains, Assess())
    obs_circ      = gen_fn_circuit(d.obs_model, (arg_domains..., latent_domains...), Assess())

    particle = ISParticle(
        proposal_circ, latent_circ, obs_circ,
        latent_addr_order, obs_addr_order,
        false # don't multiply the output scores
    )
    in_doms = input_domains(latent_circ)
    out_doms = output_domain(obs_circ)

    return PseudoMarginalizedGenFn(particle, in_doms, out_doms, output_var_addr)
end
pmdist_to_gen_fn_circuit(_, arg_domains, op::Propose) = error("Pseudo-Marginalized dists in `Propose` mode are currently not supported.")

Circuits.implement(d::PseudoMarginalizedGenFn, ::Target) = 
    CompositeComponent(
        inputs(d), outputs(d),
        (particle=d.particle,),
        (
            Input(:inputs)                   => CompIn(:particle, :args),
            CompOut(:particle, :weight)      => Output(:score),
            Input(:obs) => Output(:value)

            # TODO: remove these comments -
            # Currently we expect that there is exactly 1 observed value which is output, so this is the return value
            # CompOut(:particle, :trace => d.output_var_addr) => Output(:value)
        ), d
    )