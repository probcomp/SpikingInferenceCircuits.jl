# Change some hyperparameters for auto-normalization
# so that it works in this model.
ProbEstimates.MultAssemblySize() = 100
ProbEstimates.AutonormalizeRepeaterRate() = 5 * ProbEstimates.MaxRate()
ProbEstimates.AutonormalizationLatency() = 500

function importance_sample(prev_tr, obs_choicemap, obs_to_prop_input, proposal; fail_cnt=0)
    try
        wt = DynamicModels.use_propose_weights!()
        choices, propose_score, _ = propose(proposal, (prev_tr, obs_to_prop_input(obs_choicemap)))
        DynamicModels.done_using_propose_weights!(wt)
        t = get_args(prev_tr)[1]
        newtr, assess_score, _, _ = update(
            prev_tr, (t + 1,), (UnknownChange(),),
            merge(DynamicModels.nest_at(:steps => t + 1 => :obs, obs_choicemap), choices)
        )
        return (newtr, assess_score - propose_score)
    catch e
        (e isa InterruptException || fail_cnt > 8) && throw(e)
        @warn "Got exception when taking importance sample; will retry.  Exception: $e"
        return importance_sample(prev_tr, obs_choicemap, obs_to_prop_input, proposal; fail_cnt = fail_cnt + 1)
    end
end
function importance_samples(prev_tr, obs_cm, obs_to_prop_input, proposal, n_particles; fail_cnt=0)
    try
        samples = [importance_sample(prev_tr, obs_cm, obs_to_prop_input, proposal) for _=1:n_particles]
        trs = map(first, samples)
        logweights = map(x -> x[2], samples)

        # simulate noise from auto-normalization:
        (log_total_weight, log_normalized_weights) = ProbEstimates.normalize_weights(logweights)
        weights = log_normalized_weights .+ log_total_weight
        if (ProbEstimates.weighttype == :perfect) && !any(isnan(w) || isinf(w) for w in logweights)
            @assert !any(isnan(w) || isinf(w) for w in weights)
        end
        return zip(trs, weights) |> collect
    catch e
        (e isa InterruptException || fail_cnt > 8) && throw(e)
        @warn "Got exception when taking importance samples; will retry.  Exception: $e"
        return importance_samples(prev_tr, obs_cm, obs_to_prop_input, proposal, n_particles; fail_cnt = fail_cnt + 1)
    end
end

function average(itr)
    c = collect(itr)
    return sum(c)/length(c)
end
function log_avg(itr)
    c = collect(itr)
    return logsumexp(c) - log(length(c))
end

# Returns an iterator over pairs `(trace, obs_choicemap_for_next_timestep)`.
function get_prevtr_obs_pairs(obs_trs, get_returned_obs; n_particles=30)
    pairs = []
    for tr in obs_trs
        (first_obs_cm, rest_obs_cms) = get_returned_obs(tr)
        (unweighted_trs, _) = dynamic_model_smc(
            model, (first_obs_cm, rest_obs_cms),
            cm -> (obs_choicemap_to_vec_of_vec(cm),),
            initial_near_locopt_proposal, step_near_locopt_proposal, n_particles
        );

        for (particles, next_obs) in zip(unweighted_trs, rest_obs_cms)
            for tr in particles
                push!(pairs, (tr, next_obs))
            end
        end
    end
    return pairs
end

function get_z_estimates(prev_obs_pairs, obs_transform, proposal; n_particles, n_runs)
    z_estimates = []
    for (prevtr, obs) in prev_obs_pairs
        samples = [
            importance_samples(prevtr, obs, obs_transform, proposal, n_particles) for _=1:n_runs
        ]
        push!(z_estimates, [log_avg(wt for (tr, wt) in sample) for sample in samples])
    end
    return z_estimates
end

function gold_standard_z_estimates(prev_obs_pairs, obs_cm_to_propinput; n_particles)
    ProbEstimates.use_perfect_weights!()
    return map(only,
        get_z_estimates(prev_obs_pairs, obs_cm_to_propinput,
            step_near_locopt_proposal; n_particles, n_runs=1)
    )
end

# Currently, each estimator_spec is a pair `(step_proposal, n_particles, set_ngf_mode!)`.
# `estimator_specs` is a list of such triples.
# `set_ngf_mode!` is called to set the neural-gen-fast mode before z estimates are used.
# `obs_cm_to_propinput` maps obs choicemap -> input to proposal
# `tr_to_obs` gets the obs choicemap from a trace.
# This function outputs `(gold_standard_z_ests, z_estimates)`
# `z_estimates[p][e]` is the `e`th z estimate of the `p`th (trace, observation) pair
# and `gold_standard_z_ests[p]` is the gold-standard z estimate for the `p`th pair
function run_z_estimates_comparison(
    obs_trs, tr_to_obs, obs_cm_to_propinput, estimator_specs;
    n_particles_when_producing_prev_traces=30,
    n_particles_goldstandard=300,
    n_estimates_per_spec
)
    prev_obs_pairs = get_prevtr_obs_pairs(obs_trs, tr_to_obs, n_particles=n_particles_when_producing_prev_traces)
    gold_standard_z_ests = gold_standard_z_estimates(prev_obs_pairs, obs_cm_to_propinput, n_particles=n_particles_goldstandard)
    println("Got gold standard estimates.  Now will get estimates.")
    z_estimates = []
    for (proposal, n_particles, set_ngf_mode!) in estimator_specs
        set_ngf_mode!()
        push!(z_estimates,
            get_z_estimates(
                prev_obs_pairs, obs_cm_to_propinput, proposal;
                n_particles, n_runs=n_estimates_per_spec
            )
        )
    end
    return (gold_standard_z_ests, z_estimates)
end

# TODO: I could set this up to save data incrementally in case an error occurs
# partway through the run.
using Dates, Serialization
generate_filename(;
    directoryname=joinpath(
        Base.find_package("SpikingInferenceCircuits") |> dirname |> dirname,
        "experiments/tracking_with_occlusion/z_estimate_plot/saves"
    ),
    date=Dates.now(),
    filename=""
) = joinpath(
    directoryname,
    filename*Dates.format(date, "yyyy-mm-dd__HH-MM-SS")
)
function run_and_save_z_estimates_comparison(args...; filename=nothing, kwargs...)
    _filename =
        if isnothing(filename)
            generate_filename()
        else
            filename
        end
    ests = (gold_standard_z_ests, z_estimates) = run_z_estimates_comparison(args...; kwargs...)
    serialize(_filename, ests)
    
    return _filename
end
load_z_estimates_comparison(filename) = deserialize(filename)