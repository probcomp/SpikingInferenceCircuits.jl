function product_to_single_line(trueproduct)
#    @assert !isnan(trueproduct)
    if isnan(trueproduct)
        return NaN
    end
    if isinf(trueproduct)
        return Inf
    end
    @assert !isinf(trueproduct)
    @assert trueproduct ≥ 0.
    
    # ErlangShape = number of spikes used by timer
    T = rand(Erlang(
        TimerNSpikes()/TimerExpectedT() |> Int,
        TimerNSpikes()
    ))

    correct_rate = MultOutDenominator() * trueproduct / TimerExpectedT()
    truncated_rate = min(correct_rate, MaxMultRate())
    if correct_rate > MaxMultRate()
        truncated_value = MaxMultRate() / MultOutDenominator() * TimerExpectedT()
        @warn "Truncating value $trueproduct in multiplier output, since it would require mult rate = $(correct_rate / MultAssemblySize()) per neuron.  Truncating to value $(truncated_value)."
    end
    C = rand(Poisson(T * truncated_rate))
    return C / MultOutDenominator()
end

function normalize_weights(log_weights)
    if weighttype === :perfect
        Gen.normalize_weights(convert(Vector{Float64}, log_weights))
    else
        if any(x -> isnan(x), exp.(log_weights))
            print("displaying LOG WEIGHTS")
            display(log_weights)
        end

        if UseLowPrecisionMultiply()
            single_line_estimates = map(product_to_single_line ∘ exp, log_weights)
            log_estimate_sum = log(sum(single_line_estimates))

            return (
                log_estimate_sum,
                log.(single_line_estimates) .- log_estimate_sum
            )
        else
            return autonormalize_weights(log_weights, WeightAutonormalizationParams()...)
        end
    end
end

function autonormalize_weights(log_weights, k, speedup_factor, repeater_rate)
    rates = exp.(log_weights)
    

#    if isnan(sum(rates))
 #       return (NaN, [NaN for _ in log_weights])
  #  end
    
    if sum(rates) == 0.
        return (-Inf, [-Inf for _ in log_weights])
    end

    if sum(rates) == Inf || isnan(sum(rates))
        return (NaN, [NaN for _ in log_weights])
    end
    
    num_accumulated = 0
    num_speedups = 0
    total_time_passed = 0
    while num_accumulated < k
        time_to_repeater = rand(Exponential(1/repeater_rate))
        # Spikes before then gets a NaN value
        print("sum rates")
        print(sum(rates))
        
        spikes_before_then = rand(Poisson(sum(rates) * time_to_repeater))
        total_this_would_accumulate_to = num_accumulated + spikes_before_then
        if total_this_would_accumulate_to < k
            num_accumulated = total_this_would_accumulate_to
            num_speedups += 1
            rates = rates .* speedup_factor
            total_time_passed += time_to_repeater
        else
            num_accumulated = k
        end
    end
    @assert total_time_passed < AutonormalizationLatency() "Time passed for renormalization: $total_time_passed"
    resulting_rate_min_threshold = AutonormalizationMinResultingRate()
    @assert resulting_rate_min_threshold ≤ sum(rates) "total resulting auto-normalized rate: $(sum(rates)) [rates: $rates]"
    @assert maximum(rates) ≤ MaxRate() * MultAssemblySize() "max rate after auto-normalize: $(maximum(rates)) [rates: $rates]"

    result = (
        log(sum(rates)) - (num_speedups * log(speedup_factor)), # NOTE: THIS DOES NOT CURRENTLY FACTOR IN READ-OUT NOISE IN THE RESULTING RATES!
        log.(normalize(rates)) # we can then use a WTA to sample from the resulting rates
    )

    # this should actually not skew the distribution at all -- double check this
    perfect_result = Gen.normalize_weights(convert(Vector{Float64}, log_weights))
    @assert isapprox(result[1], perfect_result[1], atol=1e-3) "auto-normalize: $(result[1]) | Gen: $(perfect_result[1])"
    @assert isapprox(result[2], perfect_result[2], atol=1e-3) "auto-normalize: $(result[2]) | Gen: $(perfect_result[2])"

    return result
end
