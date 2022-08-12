macro compile_2timestep_proposal(initial_proposal, step_proposal, n_latents, n_obs_inputs)
    obs1_argnames = [Symbol("o1_$i") for i=1:n_obs_inputs]
    obs2_argnames = [Symbol("o2_$i") for i=1:n_obs_inputs]
    latent_argnames = [Symbol("l_$i") for i=1:n_latents]
    return quote
        # @gen (static) function __proposal($(latent_argnames...), $(obs2_argnames...))
        #     {:latents} ~ $(esc(step_proposal))($(latent_argnames...), $(obs2_argnames...))
        # end
        @gen (static) function _first_2_timestep_proposal($(obs1_argnames...), $(obs2_argnames...))
            latents = {:init => :latents} ~ $(esc(initial_proposal))($(obs1_argnames...))
            ($(latent_argnames...),) = latents
            {:steps => 1 => :latents} ~ $(esc(step_proposal))($(latent_argnames...), $(obs2_argnames...)) # __proposal($(latent_argnames...), $(obs2_argnames...))
        end
    end
end

"""
(Timesteps are 0-indexed.)
At timestep 1, this proposes the latent variables for timestep 1 and 0.
For timestep 2, 3, ..., this performs normal SMC.
"""
function deferred_dynamic_model_smc(
    model,
    (first_obs_cm, obs_cms),
    obs_cm_to_proposal_input, # obs choicemap -> arg sequence for proposal
    proposal_first_2_steps, step_proposal,
    n_particles;
    ess_threshold=Inf,
    rejuvenate=identity
)
    unweighted_traces = []
    weighted_traces = []

    function resample_rejuvenate_and_track_traces!(state)
        push!(weighted_traces, collect(zip(state.traces, state.log_weights)))

        maybe_resample!(state, ess_threshold=ess_threshold)

        for i=1:n_particles
            state.traces[i] = rejuvenate(state.traces[i])
        end

        push!(unweighted_traces, copy(state.traces))
    end

    (second_obs_cm, obs_cms) = Iterators.peel(obs_cms)
    state = Gen.initialize_particle_filter(
        model, (1,),
        Gen.merge(
            DynamicModels.nest_at(:init => :obs, first_obs_cm),
            DynamicModels.nest_at(:steps => 1 => :obs, second_obs_cm)
        ),
        proposal_first_2_steps,
        (obs_cm_to_proposal_input(first_obs_cm)..., obs_cm_to_proposal_input(second_obs_cm)...),
        n_particles
    )
    resample_rejuvenate_and_track_traces!(state)

    for (t_minus_1, o) in enumerate(obs_cms)
        t = t_minus_1 + 1
        Gen.particle_filter_step!(
            state, (t,), (UnknownChange(),),
            DynamicModels.nest_at(:steps => t => :obs, o),
            step_proposal,
            obs_cm_to_proposal_input(o)
        )
        resample_rejuvenate_and_track_traces!(state)
    end

    return (unweighted_traces, weighted_traces)
end
