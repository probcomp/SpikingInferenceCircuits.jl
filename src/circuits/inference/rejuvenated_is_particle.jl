struct RejuvenatedISParticle
    particle :: ISParticle
    rejuv    :: RejuvenationKernel
end
Circuits.inputs(r::RejuvenatedISParticle) = inputs(r.particle)
Circuits.outputs(r::RejuvenatedISParticle) = inputs(r.particle)
Circuits.implement(r::RejuvenatedISParticle, ::Target) =
    CompositeComponent(
        inputs(r), outputs(r),
        (particle=r.particle, rejuv=r.rejuv),
        (
            Input(:args) => CompIn(:particle, :args),
            Input(:obs) => CompIn(:particle, :obs),

            Input(:args) => CompIn(:rejuv, :model_args),
            Input(:obs) => CompIn(:rejuv, :obs),
            CompOut(:particle, :trace) => CompIn(:rejuv, :prev_latents),

            CompOut(:particle, :weight) => Output(:weight),
            CompOut(:rejuve, :next_latents) => Output(:trace)
        ), r
    )

maybe_add_mh_rejuv(::Nothing) = particle -> particle
maybe_add_mh_rejuv(rejuv_proposal::ImplementableGenFn) = 
    particle -> 
        RejuvenatedISParticle(particle,
            MHKernel(
                rejuv_proposal,
                particle.assess_latents, particle.assess_obs,
                particle.latent_addr_order, particle.obs_addr_order
            )
        )