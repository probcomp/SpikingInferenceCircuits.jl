function resampler_groups(indices, ready_time, n_neurons=15)
    l = 5/3 * ProbEstimates.Latency() - ready_time
    if l < 5
        l = 5
    end
    return vcat([
        [
            i == idx ? get_PP_for_length(l, n_neurons) .+ ready_time : []
            for i=1:length(indices)
        ]
        for idx in indices
    ]...)
end
function get_PP_for_length(l, n_neurons)
    spikes = []
    t = exponential(ProbEstimates.MaxRate() * n_neurons)
    while t < l
        push!(spikes, t)
        t += exponential(ProbEstimates.MaxRate() * n_neurons)
    end
    return spikes
end