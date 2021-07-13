module DynamicModels

using Gen

"""
    model = @DynamicModel(initial_latent_model, latent_step_model, obs_model, num_latent_variables)

Constructs a generative function `model` such that `model(T)` produces a trace containing
`T + 1` latent variable assignments, and `T + 1` observations.  The first latent variables
are sampled from `initial_latent_model()`, and the
latent variables at time `t` are sampled from `latent_step_model(latentsₜ₋₁)`.
Observations at time `t` are sampled from `obs_model(latentsₜ)`.

`initial_latent_model` should accept 0 arguments, and output a tuple of `num_latent_variables` latent variable
values `(x₁, ..., xₙ)`.  `latent_step_model` should accept `num_latent_variables` arguments (the latent
variables from the previous timestep), and output a tuple of `num_latent_variables` values (the
latent variables at the next timestep).  `obs_model` should accept `num_latent_variables` arguments
(the latents at the current timestep) and output a tuple of values (the observations at that timestep).

The choicemap of `model` will be such that:
- `:init => :latents` and `:init => :obs` contain the choicemaps for the first latents and observations.
- `:init => :step => t => :latents` and `:init => :step => t => :obs` contain the latent and observation model choicemaps
  for the `t`th timestep after the initial one.

Currently, `model` does not have a return value (ie. it outputs nothing), but observation values
can be easily accessed from a trace via `get_dynamic_model_obs`.
"""
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
"""
    @compile_step_proposal(step_proposal, num_latent_variables, num_obs_variables)

Converts a step proposal for a step model into a proposal compatible with a dynamic model
built from that step model using the `@DynamicModel` macro.

`step_proposal` should accept `num_latent_variables + num_obs_variables` arguments,
where the first `num_latent_variables` arguments are the latents from the previous timestep,
and the remaining `num_obs_variables` arguments are the observations from the current timestep.
`step_proposal` should trace a value at exactly the same set of addresses
traced in the `step_model` in the model it is a proposal for.
"""
macro compile_step_proposal(
    step_proposal, n_latents, n_obs_inputs
)
    prop_argnames = [Symbol("a$i") for i=1:n_latents]
    obs_argnames = [Symbol("o$i") for i=1:n_obs_inputs]
    return quote
        @gen (static) function _step_proposal(prev_tr, $(obs_argnames...))
            T = get_args(prev_tr)[1] + 1
            prev_latents = prev_tr[$(latent_addr)(T - 1)]
            ($(prop_argnames...),) = prev_latents

            {:steps => T => :latents} ~ $(esc(step_proposal))($(prop_argnames...), $(obs_argnames...))
        end
    end
end

"""
    @compile_initial_proposal(initial_proposal, n_obs_inputs)

Converts an initial proposal for the initial model used to construct a dynamic model
using the `@DynamicModel` macro into a proposal compatible with the DynamicModel.

`initial_proposal` should accept `n_obs_inputs` arguments (the observations
at the initial timestep), and should trace a value at the same set of addresses
as the initial latents model.
"""
macro compile_initial_proposal(
    initial_proposal, n_obs_inputs
)
    obs_argnames = [Symbol("o$i") for i=1:n_obs_inputs]
    return quote
        @gen (static) function _initial_proposal($(obs_argnames...))
            {:init => :latents} ~ $(esc(initial_proposal))($(obs_argnames...))
        end
    end
end

# TODO: @compile_initial_proposal

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
"""
Choicemap from the obs model at the `t`th time step.
"""
obs_choicemap(tr, t) = get_submap(get_choices(tr), obs_addr(t))
"""
Choicemap from the latent model at the `t`th time step.
(Either from the initial or step latent model.)
"""
latents_choicemap(tr, t) = get_submap(get_choices(tr), latent_addr(t))


function dynamic_model_smc(
    model,
    (first_obs_cm, obs_cms),
    obs_cm_to_proposal_input, # obs choicemap -> arg sequence for proposal
    initial_proposal, step_proposal,
    n_particles;
    ess_threshold=Inf,
    rejuvenate=identity
)
    unweighted_traces = []
    weighted_traces = []

    function resample_rejuvenate_and_track_traces!(state)
        push!(weighted_traces, collect(zip(state.traces, state.log_weights)))

        # always resample
        Gen.maybe_resample!(state, ess_threshold=ess_threshold)

        for i=1:n_particles
            state.traces[i] = rejuvenate(state.traces[i])
        end

        push!(unweighted_traces, copy(state.traces))
    end

    state = Gen.initialize_particle_filter(
        model, (0,),
        nest_at(:init => :obs, first_obs_cm),
        initial_proposal, obs_cm_to_proposal_input(first_obs_cm),
        n_particles
    )
    resample_rejuvenate_and_track_traces!(state)

    for (t, o) in enumerate(obs_cms)
        Gen.particle_filter_step!(
            state, (t,), (UnknownChange(),),
            nest_at(:steps => t => :obs, o),
            step_proposal,
            obs_cm_to_proposal_input(o)
        )
        resample_rejuvenate_and_track_traces!(state)
    end

    return (unweighted_traces, weighted_traces)
end

function nest_at(addr, submap::Gen.ChoiceMap)
    c = choicemap()
    set_submap!(c, addr, submap)
    return c
end

export @DynamicModel, @compile_step_proposal, @compile_initial_proposal
export dynamic_model_smc, get_dynamic_model_obs
export obs_choicemap, latents_choicemap

include("enumeration_bayes_filter.jl")

export EnumerationBayesFilter, enumeration_bayes_filter_from_groundtruth
export enumeration_filter_init, enumeration_filter_step

end