function product_to_single_line(trueproduct)
    # ErlangShape = number of spikes used by timer
    T = rand(Erlang(
        TimerNSpikes()/TimerExpectedT() |> Int,
        TimerNSpikes()
    ))
    C = rand(Poisson(
        T * MultOutDenominator() * trueproduct / TimerExpectedT()
    ))
    return C / MultOutDenominator()
end

function normalize_weights(log_weights::Vector{Float64})
    if weighttype === :perfect
        Gen.normalize_weights(log_weights)
    else
        single_line_estimates = map(product_to_single_line ∘ exp, log_weights)
        log_estimate_sum = log(sum(single_line_estimates))
        return (
            log_estimate_sum,
            log.(single_line_estimates) .- log_estimate_sum
        )
    end
end

# Redefine the Gen resampling function, so that it introduces approximation error
function Gen.maybe_resample!(
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