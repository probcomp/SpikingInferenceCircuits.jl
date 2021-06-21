include("model_proposal.jl")

function particle_filter(
    obs_choicemaps,
    initial_latents,
    n_particles
)
    unweighted_samples = []
    weighted_samples = []

    latent_seqs = [[initial_latents] for _=1:n_particles]
    for obs in obs_choicemaps
        chs, logweights = [importance_sample(last(l), obs) for l in latent_seqs] |> unzip
        resampled_sequences = resample_from_logweights(chs, logweights)
        
        push!(weighted_samples, (chs, logweights))
        push!(unweighted_samples, resampled)

        latents = map(choicemap_to_latent_tuple, resampled)
    end

    return (weighted_samples, unweighted_samples)
end

function importance_sample(latents, obs)
    ch, pw, _ = propose(step_proposal, (latents..., obs[:obsx], obs[:obsy]))
    aw, _ = assess(step_model, latents, merge(ch, obs))
    return (ch, aw - pw)
end

function resample_from_logweights(chs, logweights)
    probs = exp.(logweights .- logsumexp(logweights))
    println("logweights = $logweights, probs = $probs")
    resample_idxs =
        isapprox(sum(probs), 1.) ? [categorical(probs) for _=1:length(chs)] :
                   [uniform_discrete(1, length(probs)) for _=1:length(chs)]
    
    return [chs[i] for i in resample_idxs]
end

function unzip(vals)
    nvals = length(first(vals))
    return Tuple(
        [v[i] for v in vals]
        for i=1:nvals
    )
end

tuple_to_latent_choicemap((x, vx, y, vy)) = choicemap(
    (:xₜ, x), (:vxₜ, vx), (:yₜ, y), (:vyₜ, vy)
)
choicemap_to_latent_tuple(ch) = (ch[:xₜ], ch[:vxₜ], ch[:yₜ], ch[:vyₜ])
ch_to_prev_timestep(ch) = choicemap(
    (:xₜ₋₁, ch[:xₜ]), (:vxₜ₋₁, ch[:vxₜ]), (:yₜ₋₁, ch[:yₜ]), (:vyₜ₋₁, ch[:vyₜ])
)

# Comparison to particle filtering algorithm:
begin
    pick_point_near_obs(ch) = choicemap(
        (:xₜ, categorical(truncated_discretized_gaussian(ch[:obsx], 2.0, Positions()))),
        (:yₜ, categorical(truncated_discretized_gaussian(ch[:obsy], 2.0, Positions()))),
    )
    pick_vel(xₜ₋₁, xₜ, vxₜ₋₁) =
        truncate_value(
            truncate_value(xₜ - xₜ₋₁, (vxₜ₋₁ - 1):(vxₜ₋₁ + 1)),
            Vels()
        )

    function dumb_select_latent_choicemap(obs, prev_latents)
        ch = pick_point_near_obs(obs)
        ch[:vxₜ] = pick_vel(prev_latents[:xₜ₋₁], ch[:xₜ], prev_latents[:vxₜ₋₁])
        ch[:vyₜ] = pick_vel(prev_latents[:yₜ₋₁], ch[:yₜ], prev_latents[:vyₜ₋₁])
        return ch
    end

    pick_points_near_obs(obs_choicemaps) = map(pick_point_near_obs, obs_choicemaps)
    function dumb_select_latents(obs_choicemaps, initial_latents)
        latents = initial_latents
        chs = []
        for obs in obs_choicemaps
            latents = dumb_select_latent_choicemap(obs, ch_to_prev_timestep(latents))
            push!(chs, latents)
        end
        return chs
    end
end

function joint_trace_logprob(observations, latent_chs)
    latent_pairs = zip(latent_chs, Iterators.drop(latent_chs, 1))
    jointscore = 0
    for (obs, (prev_lats, new_lats)) in zip(observations, latent_pairs)
        jointscore += assess(
            step_model,
            choicemap_to_latent_tuple(prev_lats),
            merge(obs, new_lats)
        )[1]
    end
    return jointscore
end