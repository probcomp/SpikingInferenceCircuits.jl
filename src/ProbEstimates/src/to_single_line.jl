function product_to_single_line(trueproduct)
    @assert isnan(trueproduct)
    @assert !isinf(trueproduct)
    
    if trueproduct < 0
        println("trueproduct: $trueproduct ")
        return trueproduct
    end

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

function normalize_weights(log_weights)
    if weighttype === :perfect
        Gen.normalize_weights(convert(Vector{Float64}, log_weights))
    else
        if any(x -> isnan(x), exp.(log_weights))
            display(log_weights)
        end
        single_line_estimates = map(product_to_single_line ∘ exp, log_weights)
        log_estimate_sum = log(sum(single_line_estimates))
        return (
            log_estimate_sum,
            log.(single_line_estimates) .- log_estimate_sum
        )
    end
end
