#=
One current restriction is that the model must accept the previous latent values as inputs
in the same order that the proposal samples them.

Outputs the resampled traces at each step.
=#
struct SMC <: GenericComponent
    num_particles::Int

    #=
    IS particle where:
    - assess_args = previous timestep's latents
    - propose_args = previous timestep latents + current timestep obs
    =#
    is_particle::ISParticle
    # TODO: rejuventation kernels

    function SMC(np::Int, is_particle::ISParticle)
        smc = new(np, is_particle)
        @assert all(v1 == v2 for (v1, v2) in zip(values(new_latents_val(smc)), values(prev_latents_val(smc)))) "Prev latents (inputs to model) don't appear to be in same order as values sampled by proposal.  Inputs to model: $(collect(keys(prev_latents_val(smc)))); proposal sampling order: $(collect(keys(new_latents_val(smc))))."
        return smc
    end
end

SMC(num_particles::Int,  model::Gen.GenerativeFunction, proposal::Gen.GenerativeFunction, model_arg_domains, proposal_arg_domains) =
    SMC(num_particles, ISParticle(model, proposal, model_arg_domains, proposal_arg_domains))

# prev vs new will have different names, but should have the same values in the same order
# (e.g. `xₜ₋₁` vs `xₜ`)
new_latents_val(smc::SMC) = outputs(smc.is_particle)[:trace]
prev_latents_val(smc::SMC) = inputs(smc.is_particle)[:assess_args]

obs_val(smc::SMC) = inputs(smc.is_particle)[:obs]

Circuits.inputs(smc::SMC) = NamedValues(
    :initial_latents => IndexedValues(
        prev_latents_val(smc)
        for _=1:smc.num_particles
    ),
    :obs => obs_val(smc)
)
# inferred current latents for each particle  (after resampling, so they are equally weighted)
Circuits.outputs(smc::SMC) = IndexedValues(
    new_latents_val(smc)
    for _=1:smc.num_particles
)

obs_to_particle_edges(smc, sending_nodename, i) = (
    sending_nodename => CompIn(:particles => i, :obs),
    (
        Circuits.append_to_valname(sending_nodename, addr) => CompIn(:particles => i, :propose_args => addr)
        for addr in keys(obs_val(smc))
    )...
)
latents_to_particle_edges(smc, sending_nodename, i) = (
    sending_nodename => CompIn(:particles => i, :assess_args),
    (
        Circuits.append_to_valname(sending_nodename, addr) => CompIn(:particles => i, :propose_args => addr)
        for addr in keys(prev_latents_val(smc))
    )...
)

Circuits.implement(smc::SMC, ::Target) =
    CompositeComponent(
        inputs(smc), outputs(smc),
        (
            particles=IndexedComponentGroup(
                smc.is_particle for _=1:smc.num_particles
            ),
            resample=Resample(smc.num_particles, new_latents_val(smc)),
            step=Step(NamedValues(
                :latents => IndexedValues(prev_latents_val(smc) for _=1:smc.num_particles),
                :obs => obs_val(smc)
            ))
        ),
        (
            # obs -> step
            Input(:obs) => CompIn(:step, :in => :obs),

            # initial latents -> step
            Input(:initial_latents) => CompIn(:step, :in => :latents),

            Iterators.flatten( # obs step  -> particles
                obs_to_particle_edges(smc, CompOut(:step, :out => :obs), i)
                for i=1:smc.num_particles
            )...,
            Iterators.flatten( # step latents -> particles
                latents_to_particle_edges(smc, CompOut(:step, :out => :latents => i), i)
                for i=1:smc.num_particles
            )...,
            Iterators.flatten( # particle outputs -> resample
                (
                    CompOut(:particles => i, :trace) => CompIn(:resample, :traces => i),
                    CompOut(:particles => i, :weight) => CompIn(:resample, :weights => i)
                )
                for i=1:smc.num_particles
            )...,
            # recur latents to beginning
            Iterators.flatten(
                (
                    CompOut(:resample, :traces => i => newaddr) => CompIn(:step, :in => :latents => i => oldaddr)
                    for (oldaddr, newaddr) in zip(keys(prev_latents_val(smc)), keys(new_latents_val(smc)))
                )
                for i=1:smc.num_particles
            )...,

            # output resampled traces
            (
                CompOut(:resample, :traces => i) => Output(i)
                for i=1:smc.num_particles
            )...
        ),
        smc
    )