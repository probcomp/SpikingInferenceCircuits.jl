using Gen
using CairoMakie

function run_pf(obs, n)
    particles = [normal(0, 1) for _=1:n]
    logweights = [0. for _ in particles]
    resampled_indices = []
    for (t, y) in enumerate(obs)
        for (i, (x_prev, lw)) in enumerate(zip(particles, logweights))
            x = normal(x_prev, 1)
            Δlw = logpdf(normal, y, x, 1)
            lw = lw + Δlw
            particles[i] = x
            logweights[i] = lw
        end
        weights = exp.(logweights .- logsumexp(logweights))
        indices_to_resample = [i for i=1:n if weights[i] < 1/(50*n)]
        push!(resampled_indices, indices_to_resample)
        lse = logsumexp(logweights)
        for i in indices_to_resample
            j = categorical(weights)
            particles[i] = particles[j]
            logweights[i] = lse - log(n)
        end
    end
    return resampled_indices
end

ys = [1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8, 1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8, 1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8, 1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8, 1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8, 1., 2., 1.5, 2.2, 2.5, 3.0, 4.0, 4.1, 3.9, 4.0, 3.0, 2.5, 1.8]
inds = run_pf(ys[1:40], 100)
lens = collect(Iterators.flatten([(length(is), 0) for is in inds]))
lines(1:length(lens), lens)