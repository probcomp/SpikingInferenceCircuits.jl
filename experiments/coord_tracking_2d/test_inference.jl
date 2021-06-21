include("dynamic_model.jl")

@gen (static) function dumb_step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    xₜ ~ Cat(truncated_discretized_gaussian(obsx, 2.0, Positions()))
    yₜ ~ Cat(truncated_discretized_gaussian(obsy, 2.0, Positions()))
    vxₜ ~ LCat(Vels())(onehot(
        truncate_value(truncate_value(xₜ - xₜ₋₁, (vxₜ₋₁ - 1):(vxₜ₋₁ + 1)), Vels()),
        Vels()
    ))
    vyₜ ~ LCat(Vels())(onehot(
        truncate_value(truncate_value(yₜ - yₜ₋₁, (vyₜ₋₁ - 1):(vyₜ₋₁ + 1)), Vels()),
        Vels()
    ))
end
NSTEPS = 10
dm = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4)
smart_sp = @compile_step_proposal(step_proposal, 4, 2)
dumb_sp = @compile_step_proposal(dumb_step_proposal, 4, 2)
@load_generated_functions()

tr = simulate(dm, (NSTEPS,))
obss = get_dynamic_model_obs(tr)

smc_inferences = dynamic_model_smc(
    dm, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
    initial_proposal, smart_sp, 20
)

dumb_inferences = dynamic_model_smc(
    dm, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
    initial_proposal, dumb_sp, 1 # 1 particle so no the weights have no effect
)

smc_unw = smc_inferences[1];
dumb_unw = dumb_inferences[1];

avg_dist_from_ground_truth(inferred_trace_for_step, gt, T) = sum(
        dist(inferred_trace_for_step(t), gt, t)
        for t=1:T
    ) / T

function dist(tr1, tr2, t)
    x1 = tr1[:steps => t => :latents => :xₜ => :val]
    y1 = tr1[:steps => t => :latents => :yₜ => :val]
    x2 = tr2[:steps => t => :latents => :xₜ => :val]
    y2 = tr2[:steps => t => :latents => :yₜ => :val]
    return abs(x1 - x2) + abs(y1 - y2)
end

function avg_dists_from_ground_truth(n_steps, n_traces, n_runs_per_trace, n_particles)
    trs = [simulate(dm, (n_steps,)) for _=1:n_traces]
    obsss = map(get_dynamic_model_obs, trs)
    smart_unweighted_inferences = [
        [
            dynamic_model_smc(
                dm, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
                initial_proposal, smart_sp, n_particles
            )[1]
            for _=1:n_runs_per_trace
        ]
        for obss in obsss
    ]
    dumb_unweighted_inferences = [
        [
            dynamic_model_smc(
                dm, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
                initial_proposal, dumb_sp, 1 # 1 particle so no the weights have no effect
            )[1]
            for _=1:n_runs_per_trace
        ]
        for obss in obsss
    ]

    smart_avg_dists = [
            avg_dist_from_ground_truth(
                t -> unw[t + 1][1], tr, n_steps
            )
                for (unw_for_tr, tr) in zip(
                    smart_unweighted_inferences, trs
                )
                    for unw in unw_for_tr
    ]

    dumb_avg_dists = [
        avg_dist_from_ground_truth(
            t -> unw[t + 1][1], tr, n_steps
        )
            for (unw_for_tr, tr) in zip(
                dumb_unweighted_inferences, trs
            )
                for unw in unw_for_tr
    ]

    return (dumb_avg_dists, smart_avg_dists)
end
function mean(vals)
    c = collect(vals)
    return sum(c)/length(c)
end

# As a measure of inference quality (other than trace probability),
# we can check the average cartesian distance between the inferred
# object position and the true object position.
# (This seems like an important metric for, e.g., how likely it is
# an animal will successfully grab it's prey.)
# s_dist = avg_dist_from_ground_truth(
#     smc_unw[10][1], tr
# )
# d_dist = avg_dist_from_ground_truth(
#     dumb_unw[10][1], tr
# )

# (s_dist, d_dist)

# (dumb_avg_dists, smart_avg_dists) = avg_dists_from_ground_truth(5, 50, 5, 4)

# (mean(dumb_avg_dists), mean(smart_avg_dists))