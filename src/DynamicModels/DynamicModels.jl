module DynamicModels

using Gen

macro DynamicModel(
    initial_latent_model,
    latent_step_model,
    obs_model,
    n_latents
)
    prev_latent_names = [Symbol("prev_latent$i") for i=1:n_latents]
    latent_names = [Symbol("latent$i") for i=1:n_latents]
    return quote
        @gen (static) function initial_step()
            latents ~ $(esc(initial_latent_model))()
            ($(latent_names...),) = latents
            obs ~ $(esc(obs_model))($(latent_names...))
            return latents
        end

        @gen (static) function take_step(t, prev_latents)
            ($(prev_latent_names...),) = prev_latents
            latents ~ $(esc(latent_step_model))($(prev_latent_names...))
            ($(latent_names...),) = latents
            obs ~ $(esc(obs_model))($(latent_names...))
            return latents
        end

        @gen (static) function dynamic_model(T)
            init ~ initial_step()
            steps ~ Unfold(take_step)(T, init)
        end
    end
end

obs_addr(t)    = t == 0 ? :init => :obs     : :steps => t => :obs
latent_addr(t) = t == 0 ? :init => :latents : :steps => t => :latents
macro compile_step_proposal(
    step_proposal, n_latents, n_obs_inputs
)
    prop_argnames = [Symbol("a$i") for i=1:n_latents]
    obs_argnames = [Symbol("o$i") for i=1:n_obs_inputs]
    return quote
        @gen (static) function _step_proposal(prev_tr, $(obs_argnames...))
            T = get_args(prev_tr)[1] + 1
            prev_latents = tr[latent_addr(T - 1)]
            ($(prop_argnames...),) = prev_latents

            {:steps => T => :latents} ~ $(esc(step_proposal))($(prop_argnames...), $(obs_argnames...))
        end
    end
end

"""
Given a trace `tr` from a dynamic model,
returns the choicemaps produced by the observation model
at each timestep.

In particular, returns `(initial_obs, subsequent_obs)`
where `initial_obs` is the observation from the first timestep,
and `subsequent_obs` is a vector s.t. `subsequent_obs[t]` gives
the obs for timestep `t`.
"""
get_dynamic_model_obs(tr) = (
    get_submap(get_choices(tr), :init => :obs),
    [
        get_submap(get_choices(tr), :steps => t => :obs)
        for t=1:get_args(tr)[1]
    ]
)

function dynamic_model_smc(
    model,
    (first_obs_cm, obs_cms),
    obs_cm_to_proposal_input, # obs choicemap -> arg sequence for proposal
    initial_proposal, step_proposal,
    n_particles
)
    unweighted_traces = []
    weighted_traces = []

    function resample_and_track_traces!(state)
        push!(weighted_traces, collect(zip(state.traces, state.log_weights)))

        # always resample
        Gen.maybe_resample!(state, ess_threshold=Inf)

        push!(unweighted_traces, copy(state.traces))
    end

    state = Gen.initialize_particle_filter(
        model, (0,),
        nest_at(:init => :latents, first_obs_cm),
        initial_proposal, obs_cm_to_proposal_input(first_obs_cm),
        n_particles
    )
    resample_and_track_traces!(state)

    for (t, o) in enumerate(obs_cms)
        Gen.particle_filter_step!(
            state, (t,), (UnknownChange(),),
            nest_at(:steps => t => :obs, o),
            step_proposal,
            obs_cm_to_proposal_input(o)
        )
        resample_and_track_traces!(state)
    end

    return (unweighted_traces, weighted_traces)
end

function nest_at(addr, submap::Gen.ChoiceMap)
    c = choicemap()
    set_submap!(c, addr, submap)
    return c
end

export @DynamicModel, @compile_step_proposal, dynamic_model_smc, get_dynamic_model_obs

end