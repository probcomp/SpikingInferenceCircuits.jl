module DynamicModels
using Gen
include("prev_timestep_mh.jl")

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

addr_for_timestep(t) = t == 0 ? :init : :steps => t
obs_addr(t)          = t == 0 ? :init => :obs     : :steps => t => :obs
latent_addr(t)       = t == 0 ? :init => :latents : :steps => t => :latents
"""
    @compile_step_proposal(step_proposal, num_latent_variables, num_obs_variables)
    @compile_step_proposal(step_proposal, obs_proposal, num_latent_variables, num_obs_variables)

Converts a step proposal for a step model into a proposal compatible with a dynamic model
built from that step model using the `@DynamicModel` macro.

`step_proposal` should accept `num_latent_variables + num_obs_variables` arguments,
where the first `num_latent_variables` arguments are the latents from the previous timestep,
and the remaining `num_obs_variables` arguments are the observations from the current timestep.
`step_proposal` should trace a value at exactly the same set of addresses
traced in the `step_model` in the model it is a proposal for.

`step_proposal` proposes to the latent variables at the new timestep.
If `obs_proposal` is provided, it proposes to any variables in the observation model which are not
directly observed.  If `obs_proposal` is provided, it should accept as argument
`(latent_vars_at_time_T..., obs_vars_at_time_T...)`, where `latent_vars_at_time_T`
are the values proposed to by the `step_proposal`.
If `obs_proposal` is provided, then `step_proposal` must return a tuple of the
latents used by `obs_proposal`.
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
macro compile_step_proposal(
    step_proposal, obs_proposal, n_latents, n_obs_inputs
)
    prop_argnames = [Symbol("a$i") for i=1:n_latents]
    latents_argnames = [Symbol("l$i") for i=1:n_latents]
    obs_argnames = [Symbol("o$i") for i=1:n_obs_inputs]
    return quote
        @gen (static) function __proposal($(prop_argnames...), $(obs_argnames...))
            latents = {:latents} ~ $(esc(step_proposal))($(prop_argnames...), $(obs_argnames...))
            ($(latents_argnames...),) = latents
            {:obs} ~ $(esc(obs_proposal))($(latents_argnames...), $(obs_argnames...))
        end
        @gen (static) function _step_proposal(prev_tr, $(obs_argnames...))
            T = get_args(prev_tr)[1] + 1
            prev_latents = prev_tr[$(latent_addr)(T - 1)]
            ($(prop_argnames...),) = prev_latents
            {:steps => T} ~ __proposal($(prop_argnames...), $(obs_argnames...))
        end
    end
end


"""
    @compile_initial_proposal(initial_proposal, n_obs_inputs)
    @compile_initial_proposal(initial_proposal, obs_proposal, n_latents, n_obs_inputs)

Converts an initial proposal for the initial model used to construct a dynamic model
using the `@DynamicModel` macro into a proposal compatible with the DynamicModel.

`initial_proposal` should accept `n_obs_inputs` arguments (the observations
at the initial timestep), and should trace a value at the same set of addresses
as the initial latents model.

If `obs_proposal` is provided, this is used to propose to any variables
in the observation model which are not observed.  It accepts as arguments
`(latent_variables_at_time_0..., obs_inputs...)`, where `latent_variables_at_time_0`
is the `n_latents`-long return value from `initial_proposal` (carrying the latent
variables returned by the initial proposal).  If `obs_proposal` is provided,
then the `step_proposal` must return a tuple of the latent variables used by `obs_proposal`.
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

function get_args_for_rejuv_from_trace(trace)
    T = get_args(trace)[1]
    latent_ret = trace[latent_addr(T)]
    obs_ret    = trace[obs_addr(T)]
    return (latent_ret..., obs_ret...)
end
macro compile_rejuvenation_proposal(kernel, n_latents, n_obs)
    latent_argnames = [Symbol("a$i") for i=1:n_latents]
    obs_argnames = [Symbol("o$i") for i=1:n_obs]
    argnames = [latent_argnames..., obs_argnames...]

    return quote
        @gen (static) function init_rejuv_proposal(trace)
            ($(argnames...),) = $(get_args_for_rejuv_from_trace)(trace)
            {:init => :latents} ~ $(esc(kernel))($(argnames...))
        end
        @gen (static) function step_rejuv_proposal(trace)
            ($(argnames...),) = $(get_args_for_rejuv_from_trace)(trace)
            T = get_args(trace)[1]
            {:steps => T => :latents} ~ $(esc(kernel))($(argnames...))
        end
        
        function rejuvenate_trace(trace)
            proposal = get_args(trace)[1] == 0 ? init_rejuv_proposal : step_rejuv_proposal
            newtr, _ = $(last_timestep_mh)(trace, proposal, ())
            return newtr
        end

        rejuvenate_trace
    end
end

macro compile_initial_proposal(
    initial_proposal, obs_proposal, n_latents, n_obs_inputs
)
    latents_argnames = [Symbol("l$i") for i=1:n_latents]
    obs_argnames = [Symbol("o$i") for i=1:n_obs_inputs]
    return quote
        @gen (static) function __proposal($(obs_argnames...))
            latents = {:latents} ~ $(esc(initial_proposal))($(obs_argnames...))
            ($(latents_argnames...),) = latents
            {:obs} ~ $(esc(obs_proposal))($(latents_argnames...), $(obs_argnames...))
        end
        @gen (static) function _initial_proposal($(obs_argnames...))
            {:init} ~ __proposal($(obs_argnames...))
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

        maybe_resample!(state, ess_threshold=ess_threshold)

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
        _particle_filter_step!(
            state, (t,), (UnknownChange(),),
            nest_at(:steps => t => :obs, o),
            step_proposal,
            obs_cm_to_proposal_input(o)
        )
        resample_rejuvenate_and_track_traces!(state)
    end

    return (unweighted_traces, weighted_traces)
end
function _particle_filter_step!(state::Gen.ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap, proposal::Gen.GenerativeFunction, proposal_args::Tuple) where {U}
    num_particles = length(state.traces)
    for i=1:num_particles
        (prop_choices, prop_weight, _) = propose(proposal, (state.traces[i], proposal_args...))
        constraints = merge(observations, prop_choices)
        # println("CONSTRAINTS:")
        # display(constraints)
        (state.new_traces[i], up_weight, _, disc) = update(state.traces[i], new_args, argdiffs, constraints)
        @assert isempty(disc)
        state.log_weights[i] += up_weight - prop_weight
    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return nothing
end

function nest_at(addr, submap::Gen.ChoiceMap)
    c = choicemap()
    set_submap!(c, addr, submap)
    return c
end

maybe_resample!(state; ess_threshold) = Gen.maybe_resample!(state; ess_threshold)
resample(traces, logweights, n_samples) = [traces[categorical(normalize(exp.(logweights)))] for _=1:n_samples]

export @DynamicModel, @compile_step_proposal, @compile_initial_proposal
export dynamic_model_smc, get_dynamic_model_obs
export obs_choicemap, latents_choicemap
export @compile_rejuvenation_proposal

include("enumeration_bayes_filter.jl")

export EnumerationBayesFilter, enumeration_bayes_filter_from_groundtruth
export enumeration_filter_init, enumeration_filter_step

include("particle_gibbs.jl")
export single_step_particle_gibbs_rejuv_kernel, single_step_particle_gibbs

# SMC simulations, but where the proposed values are fixed ahead-of-time by the user
include("controlled_smc.jl")
export predetermined_dynamic_model_smc

end