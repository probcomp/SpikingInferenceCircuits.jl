# ProbEstimates can overwrite this if needed to enable the noise model
with_proposal_weighttype(f) = f()
function controlled_initialize_particle_filter(model::GenerativeFunction{T,U}, model_args::Tuple,
    observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple,
    num_particles::Int, prop_choices_for_partilces
) where {T,U}
    @assert length(prop_choices_for_partilces) == num_particles
    traces = Vector{Any}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    for (i, prop_choices) in enumerate(prop_choices_for_partilces)
        (prop_tr, prop_weight) = with_proposal_weighttype(() -> generate(proposal, proposal_args, prop_choices))
        # I think it is important to use `get_choices(prop_tr)` rather than `prop_choices` when updating the model trace,
        # to make sure the right bookkeeping for the recip prob estimates occurs
        (traces[i], model_weight) = generate(model, model_args, merge(observations, get_choices(prop_tr)))
        log_weights[i] = model_weight - prop_weight
    end
    Gen.ParticleFilterState{U}(traces, Vector{U}(undef, num_particles),
        log_weights, 0., collect(1:num_particles))
end
function extend_trace_controlled(
    prev_model_trace,  prop_choices,
    (new_args, argdiffs, new_observations, proposal, proposal_args)
)

    (prop_tr, prop_weight) = with_proposal_weighttype(() -> generate(proposal, (prev_model_trace, proposal_args...), prop_choices))
    # I think it is important to use `get_choices(prop_tr)` rather than `prop_choices` when updating the model trace,
    # to make sure the right bookkeeping for the recip prob estimates occurs
    prop_choices = get_choices(prop_tr)

    # computing the new trace via update
    constraints = merge(new_observations, prop_choices)

    # if new_args[1] == 2
    #     addr = :steps => 2 => :obs => :yᶜₜ => :val
    #     println("constraints[$addr] = $(constraints[addr])")
    #     println("pre_updated_trace has value at address: $(has_value(get_choices(prev_model_trace), addr))")
    #     println("argdiffs: $argdiffs")
    # end

    (new_model_trace, log_model_weight, _, discard) = update(
        prev_model_trace, new_args,
        argdiffs, constraints
    )
    # if new_args[1] == 2
    #     println("new_model_trace[$addr] = $(get_choices(new_model_trace)[addr])")
    # end

    if !isempty(discard)
        @error("Can only extend the trace with random choices, not remove them.")
        error("Invalid extend_trace_controlled")
    end

    log_weight = log_model_weight - prop_weight
    return (new_model_trace, log_weight)
end
function controlled_particle_filter_step!(
    state::Gen.ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple,
    proposal_choicemaps
) where {U}
    @assert length(proposal_choicemaps) == length(state.traces)
    args = (new_args, argdiffs, observations, proposal, proposal_args)
    num_particles = length(state.traces)
    log_incremental_weights = Vector{Float64}(undef, num_particles) 
    for (i, prop_choices) in enumerate(proposal_choicemaps)
        (state.new_traces[i], log_weight) = extend_trace_controlled(state.traces[i], prop_choices, args)
        log_incremental_weights[i] = log_weight
        state.log_weights[i] += log_weight
    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return (log_incremental_weights,)
end

function predetermined_dynamic_model_smc(
    model,
    (first_obs_cm, obs_cms),
    obs_cm_to_proposal_input, # obs choicemap -> arg sequence for proposal
    initial_proposal, step_proposal,
    n_particles,
    (first_proposed_cm, rest_proposed_cms);
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

    state = controlled_initialize_particle_filter(
        model, (0,),
        nest_at(:init => :obs, first_obs_cm),
        initial_proposal, obs_cm_to_proposal_input(first_obs_cm),
        n_particles, first_proposed_cm
    )
    resample_rejuvenate_and_track_traces!(state)

    for (t, (o, pc)) in enumerate(zip(obs_cms, rest_proposed_cms))
        controlled_particle_filter_step!(
            state, (t,), (UnknownChange(),),
            nest_at(:steps => t => :obs, o),
            step_proposal,
            obs_cm_to_proposal_input(o), pc
        )
        resample_rejuvenate_and_track_traces!(state)
    end

    return (unweighted_traces, weighted_traces)
end