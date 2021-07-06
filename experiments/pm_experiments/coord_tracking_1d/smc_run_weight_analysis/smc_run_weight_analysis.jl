function get_noisy_weight_statistics(
    dynamic_model,
    step_obs_model_pairs,
    latencies,
    compute_statistic;
    n_samples_per_mode_per_transition
)
    # 1. Do a run
    ProbEstimates.use_perfect_weights!()
    inference_results = do_smc_run(dynamic_model)

    # get triples (true_weight, prev_trace, updated_trace) for every transition
    # done by any particle during inference
    transitions_encountered_during_inference = extract_transitions(inference_results)

    # 2. For each obs model:
    #     For each latency:
    #      get a bunch of samples of weight estimates this model + latency would produce
    noisy_weight_samples = Dict(
        (model, noisemode) => [
            (
                trueweight,
                get_weight_samples(
                    (step_model, obs_model), (prev_trace, updated_trace), noisemode;
                    n_samples=n_samples_per_mode_per_transition
                )
            )
            for (trueweight, prev_trace, updated_trace) in transitions_encountered_during_inference
        ]
        for (step_model, obs_model) in step_obs_model_pairs
            for noisemode in (
                :perfect,
                ((:noisy, lat) for lat in latencies)...
            )
    )
    
    mode_transition_idx_to_weightsamples = Dict(
        # ith transition encountered during inference
        (key, i) => [
            vec_of_trueweight_sample_pair[i][2]
            for vec_of_trueweight_sample_pair in vec_of_vecs_of_trueweight_sample_pair
        ]
        for (key, vec_of_vecs_of_trueweight_sample_pair) in noisy_weight_samples
            for i=1:length(first(vec_of_vecs_of_trueweight_sample_pair))
    )
    transition_idx_to_trueweight = [
        trueweight
        for (trueweight, _) in first(first(values(noisy_weight_samples)))
    ]

    # 3. Compute variance (or other statistic) of weights in each case.  (I think we expect
    # stddev/value to be constant across the different weight values, so maybe use this.)
    noisy_weight_statistics = Dict(
        (key, i) => compute_statistic(weightsamples, transition_idx_to_trueweight[i])
        for ((key, i), weightsamples) in mode_transition_idx_to_weightsamples
    )

    mean_weight_statistics = Dict(
        key => sum(noisy_weight_statistics[(key, i)])/n_transitions
        for key in unique(key for (key, _) in keys(noisy_weight_statistics))
    )

    return mean_weight_statistics
end

function do_smc_run(model)
    tr = simlulate(model)
    obss = get_dynamic_model_obs(tr)
    return dynamic_model_smc(
        model, obss, cm -> (cm[:obsx => :val],),
        proposal_init, proposal_step, n_particles; ess_threshold=n_particles/2
    )
end
# get multiple samples of the weights for each transition
# output is a vector of n_samples vectors of pairs (trueweight, weightsample)
# for each transition encountered during inference for each particle
function get_weight_samples(
    (resampled_traces, weighted_pre_resample_traces),
    model, noisemode; n_samples
)
    original_mode = establish_noisemode!(noisemode)
    samples = [
        get_weight_sample(resampled_traces, weighted_pre_resample_traces, model)
        for _=1:n_samples
    ]
    reset_noisemode!(original_mode)
    return samples
end
# get a sample for each weight for each step which occurred during inference
# output is a vector of pairs (trueweight, weightsample)
get_weights_sample(resampled_traces, weighted_pre_resample_traces, model) =
    [
        (weight, get_trace_update_weight_sample(tr, updated_tr))

        for (prev_traces, weights_and_updated_traces) in  zip(
            resampled_traces, # first elem = first resampled trace
            Iterators.skip(1, weighted_pre_resample_traces) # first elem = proposed from 
        )
            for (tr, (weight, updated_tr)) in zip(prev_traces, weights_and_updated_traces)
    ]
# get a sample of the weight for a given transition
function get_trace_update_weight_sample(tr, updated_tr)
    newT = get_args(updated_tr)[1]
    force_propose_noisemode!()
    propose_weight, _ = assess(step_proposal, (tr,),
        nest_at(
            :steps => newT, 
            get_submap(get_choices(updated_tr), :steps => newT)
        )
    )

    force_assess_noisemode!()
    # TODO: this isn't quite going to work!
    _, assess_weight, _, _ = update(tr, (get_args(tr)[1] + 1,), (UnknownChange(),), proposed_choices)

    return exp.(assess_weight - propose_weight)
end

compute_std_over_weight((true_weight, weight_samples)) = stddev(weight_samples, expectation=true_weight)/true_weight
stddev(args...; kwargs...) = sqrt(var(args..., kwargs...))
var(samples; expectation=sum(samples)/length(samples)) =
    sum((s - expectation)^2 for s in samples)/length(samples)