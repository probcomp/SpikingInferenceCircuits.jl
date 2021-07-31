import DynamicModels

# Redefine the DynamicModels resampling functions so that they introduces approximation error
function DynamicModels.resample(traces, logweights, n_samples)
    (log_total_weight, log_normalized_weights) = normalize_weights(logweights)
    weights = exp.(log_normalized_weights)
    return [traces[categorical(normalize(weights))] for _=1:n_samples]
end

function DynamicModels.maybe_resample!(
    state::Gen.ParticleFilterState{U};
    ess_threshold::Real=length(state.traces)/2, verbose=false
) where {U}
    num_particles = length(state.traces)

    (log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    
    ess = Gen.effective_sample_size(log_normalized_weights)
    do_resample = ess < ess_threshold
    if verbose
        println("effective sample size: $ess, doing resample: $do_resample")
    end
    if do_resample
        weights = exp.(log_normalized_weights)
        Distributions.rand!(Distributions.Categorical(weights / sum(weights)), state.parents)
        state.log_ml_est += log_total_weight - log(num_particles)
        for i=1:num_particles
            state.new_traces[i] = state.traces[state.parents[i]]
            state.log_weights[i] = 0.
        end

        # swap references
        tmp = state.traces
        state.traces = state.new_traces
        state.new_traces = tmp
    end
    return do_resample
end

function DynamicModels.use_propose_weights!()
    if weight_type() === :noisy
        use_only_recip_weights!()
        return :noisy
    else
        return weight_type()
    end
end
DynamicModels.done_using_propose_weights!(weight_type) = set_weighttype_to!(weight_type)
