"""
    SMCStep(
        step_model_bundle               :: GenFnWithInputDomains,
        obs_model_bundle                :: GenFnWithInputDomains,
        step_proposal_bundle            :: GenFnWithInputDomains,
        latent_var_addrs_for_obs        :: Vector               ,
        obs_addr_order                  :: Vector               ,
        num_particles                   :: Int
    )

A circuit for performing an SMC step under the given step model and proposal.
Performs the operation: "draw `num_particles` importance samples,
then resample according to the importance weights, and output the resampled traces".

``obs_addr_order`` gives the order in which observations should be fed into the proposal
after the previous timesteps' latents.
`latent_var_addrs_for_obs` gives the order in which latent variables should be input into the observation
model.
"""
struct SMCStep <: Circuits.GenericComponent
    num_particles                   :: Int
    is_particle                     :: ISParticle
end
SMCStep(
    step_model_bundle               :: GenFnWithInputDomains,
    obs_model_bundle                :: GenFnWithInputDomains,
    step_proposal_bundle            :: GenFnWithInputDomains,
    latent_var_addrs_for_obs        :: Vector               ,
    obs_addr_order                  :: Vector               ,
    num_particles                   :: Int
) = SMCStep(
        num_particles,
        ISParticle(
            step_proposal_bundle, step_model_bundle, obs_model_bundle,
            latent_var_addrs_for_obs, obs_addr_order
        )
    )

# TODO: prev_latents_val, new_latents_val, obs_val
prev_latents_val(s::SMCStep) = inputs(s.is_particle)[:args]
obs_val(s::SMCStep) = inputs(s.is_particle)[:obs]
all_new_latents_val(s::SMCStep) = outputs(s.is_particle)[:trace]

Circuits.inputs(s::SMCStep) = NamedValues(
    :prev_latents => IndexedValues(prev_latents_val(s) for _=1:s.num_particles),
    :obs => obs_val(s)
)
Circuits.outputs(s::SMCStep) = IndexedValues(all_new_latents_val(s) for _=1:s.num_particles)

Circuits.implement(s::SMCStep, ::Target) =
    CompositeComponent(
        inputs(s), outputs(s), (
            particles=IndexedComponentGroup(s.is_particle for _=1:s.num_particles),
            resample=Resample(s.num_particles, all_new_latents_val(s))
        ), Iterators.flatten(
            (
                Input(:prev_latents) => CompIn(:particles => i, :args),
                Input(:obs) => CompIn(:particles => i, :obs),
                CompOut(:particles => i, :trace) => CompIn(:resample, :traces => i),
                CompOut(:particles => i, :weight) => CompIn(:resample, :weights => i),
                CompOut(:resample, :traces => i) => Output(i)
            )
            for i=1:s.num_particles
        ), s
    )

"""
    RecurrentSMCStep(step::SMCStep, latent_var_addrs_for_recurrence)

A `SMCStep` which automatically recurs latents sampled at the previous timestep back to the beginning.
The resulting behavior is that at first the circuit must receive an initial assignment to the latents
and an initial observation, and will output an initial
collection of particles for updated latents at the first timestep.
At each subsequent timesteup, an observation should be input, and the circuit will output
a collection of particles for updated latents (updating the pervious cloud of latents).
The particle clouds are unweighted (ie. we output the particles which occur
after resampling.)

``latent_var_addrs_for_recurrence`` gives the subset of the addresses traced in the model / proposal
which are latent variables (which should be input to the model/proposal at the next timestep),
and the order in which they should be input.

The implementation of this circuit includes a `STEP` unit to ensure that all values
are synchronized when a timestep passes.  Note that there may be a required delay between
when observations for timesteps are received to maintain high probability of circuit success.
"""
struct RecurrentSMCStep <: GenericComponent
    step                            :: SMCStep
    latent_var_addrs_for_recurrence :: Vector

    RecurrentSMCStep(s::SMCStep, lvar) = new(s, lvar)
end
Circuits.inputs(s::RecurrentSMCStep) = NamedValues(
    :initial_latents => inputs(s.step)[:prev_latents],
    :obs => obs_val(s.step)
)
Circuits.outputs(s::RecurrentSMCStep) = outputs(s.step)

Circuits.implement(s::RecurrentSMCStep, ::Target) =
    CompositeComponent(
        inputs(s), outputs(s),
        (
            smcstep=s.step,
            timestep=Step(NamedValues(
                :latents => IndexedValues(prev_latents_val(s.step) for _=1:s.step.num_particles),
                :obs => obs_val(s.step)
            ))
        ),
        (
            # Inputs --> timestep
            Input(:initial_latents) => CompIn(:timestep, :in => :latents),
            Input(:obs) => CompIn(:timestep, :obs),
            
            # Timestep --> SMCStep
            CompOut(:timestep, :out => :obs) => CompIn(:smcstep, :obs),
            CompOut(:timestep, :out => :latents) => CompIn(:smcstep, :prev_latents),

            ( # Recur outputted latents back into the timestep unit
                CompOut(:smcstep, i => new_latent_addr) => CompIn(:timestep, i => prev_latent_addr)
                for i=1:s.step.num_particles
                    for (new_latent_addr, prev_latent_addr) in zip(
                        s.latent_var_addrs_for_recurrence, keys(inputs(s.step)[:prev_latents])
                    )
            )...,

            # SMCStep --> Output
            (
                CompOut(:smcstep, i) => Output(i)
                for i=1:s.step.num_particles
            )...
        ), s
    )