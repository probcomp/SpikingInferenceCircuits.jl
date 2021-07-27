using Gen, Distributions
using DynamicModels

includet("model_proposal.jl")
includet("naive_inference.jl")

NSTEPS = 10
model = @DynamicModel(initial_latent_model, step_latent_model, obs_model_direct, 2)
proposal_init = @compile_initial_proposal(initial_proposal, 1)
proposal_step = @compile_step_proposal(step_proposal, 2, 1)
naive_proposal_init = @compile_initial_proposal(naive_initial_proposal, 1)
naive_proposal_step = @compile_step_proposal(naive_step_proposal, 2, 1)
@load_generated_functions()

# Get a trace to run inference on, and extract the observations from that trace
tr = simulate(model, (NSTEPS,))
obss = get_dynamic_model_obs(tr)

# Run inference using Gen floating-point score calculations
ProbEstimates.use_perfect_weights!()

do_smart_inference(obss, n_particles) = dynamic_model_smc(
    model, obss, cm -> (cm[:obsx => :val],),
    proposal_init, proposal_step, n_particles#; ess_threshold=n_particles/2
)

do_naive_inference(obss) = dynamic_model_smc(
    model, obss, cm -> (cm[:obsx => :val],),
    naive_proposal_init, naive_proposal_step, 1;
    ess_threshold=-Inf
)

avg_dist_from_ground_truth(inferred_trace_for_step, gt, T) = sum(
        dist(inferred_trace_for_step(t), gt, t)
        for t=1:T
    ) / T
dist(tr1, tr2, t) = dist(
    tr1[pairnest(DynamicModels.latent_addr(t), :xₜ => :val)],
    tr2[pairnest(DynamicModels.latent_addr(t), :xₜ => :val)]
)
dist(x1, x2) = abs(x1 - x2)
pairnest(p1, p2) = p1 => p2; pairnest(p1::Pair, p2) = pairnest(p1.first, pairnest(p1.second, p2))

function avg_dists_from_ground_truth(n_steps, n_traces, n_runs_per_trace, n_particles)
    trs = [simulate(model, (n_steps,)) for _=1:n_traces]
    obsss = map(get_dynamic_model_obs, trs)
    smart_unweighted_inferences = [
        [do_smart_inference(obss, n_particles)[1] for _=1:n_runs_per_trace]
        for obss in obsss
    ]
    dumb_unweighted_inferences = [
        [do_naive_inference(obss)[1] for _=1:n_runs_per_trace]
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

(dumb_avg_dists, smart_avg_dists) = avg_dists_from_ground_truth(10, 30, 30, 10); (mean(dumb_avg_dists), mean(smart_avg_dists))