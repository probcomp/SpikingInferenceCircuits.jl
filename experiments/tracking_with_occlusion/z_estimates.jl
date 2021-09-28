# I think this version of `importance_samples` would work, but I am not 100% sure whether it will
# interact with NeuralGen-Fast properly, so I will manually write out the `propose` and `update` calls.
# function importance_samples(prev_tr, obs_cm, obs_to_prop_input, proposal, n_particles)
#     t = get_args(prev_tr)[1]
#     trace_translator = Gen.SimpleExtendingTraceTranslator((t+1,), (UnknownChange(),), obs_cm, proposal, (obs_to_prop_input,))
#     return [trace_translator(prev_tr) for _=1:n_particles]
# end
function importance_sample(prev_tr, obs_choicemap, obs_to_prop_input, proposal)
    choices, propose_score, _ = propose(proposal, (prev_tr, obs_to_prop_input(obs_choicemap)))
    t = get_args(prev_tr)[1]
    newtr, assess_score, _, _ = update(
        prev_tr, (t + 1,), (UnknownChange(),),
        merge(DynamicModels.nest_at(:steps => t + 1 => :obs, obs_choicemap), choices)
    )
    return (newtr, assess_score - propose_score)
end
importance_samples(prev_tr, obs_cm, obs_to_prop_input, proposal, n_particles) = [importance_sample(prev_tr, obs_cm, obs_to_prop_input, proposal) for _=1:n_particles]

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

gold_standard_z_estimates(prev_obs_pairs, obs_cm_to_propinput; n_particles) =
    map(only,
        get_z_estimates(prev_obs_pairs, obs_cm_to_propinput,
            step_near_locopt_proposal; n_particles, n_runs=1)
    )

# Currently, each estimator_spec is a pair `(step_proposal, n_particles)`;
# estimator_specs is a list of such pairs
# obs transform maps obs choicemap -> input to proposal
# tr_to_obs gets the obs choicemap from a trace.
#= Outputs `(gold_standard_z_ests, z_estimates)`
`z_estimates[p][e]` is the `e`th z estimate of the `p`th (trace, observation) pair
and `gold_standard_z_ests[p]` is the gold-standard z estimate for the `p`th pair
=#
function z_estimates_comparison(
    obs_trs, tr_to_obs, obs_cm_to_propinput, estimator_specs;
    n_particles_when_producing_prev_traces=30,
    n_particles_goldstandard=300,
    n_estimates_per_spec
)
    prev_obs_pairs = get_prevtr_obs_pairs(obs_trs, tr_to_obs, n_particles=n_particles_when_producing_prev_traces)
    gold_standard_z_ests = gold_standard_z_estimates(prev_obs_pairs, obs_cm_to_propinput, n_particles=n_particles_goldstandard)
    z_estimates = [
        get_z_estimates(prev_obs_pairs, obs_cm_to_propinput, proposal; n_particles, n_runs=n_estimates_per_spec)
        for (proposal, n_particles) in estimator_specs
    ]
    return (gold_standard_z_ests, z_estimates)
end

function plot_z_estimate_comparison(gold_standard_z_ests, z_estimates, labels)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="Gold-Standard Log(P[yₜ | xₜ₋₁]) Estimate", ylabel="Log(P[yₜ | xₜ₋₁]) Estimate")
    plots = []
    for (label, ests) in zip(labels, z_estimates)
        gold_standard_with_repeats = (
            [[gs for _ in estimates] for (gs, estimates) in zip(gold_standard_z_ests, ests)]
                |> Iterators.flatten |> collect
        )
        ests_flat = ests |> Iterators.flatten |> collect
        push!(plots,
            scatter!(ax, gold_standard_with_repeats, ests_flat)
        )
    end
    m = minimum(v for v in gold_standard_z_ests if !isinf(v)); M = maximum(gold_standard_z_ests)
    yx = lines!(ax, [Point2f0(m, m), Point2f0(M, M)])
    Legend(f[1, 2], [plots..., yx], [labels..., "y=x (0-variance z-estimate)"])
    return f
end

### Script to run this

specs = [(step_prior_proposal, 10), (step_near_locopt_proposal, 10)];
labels = ["Prior Proposal (10 particles)", "Nearly Locally Optimal Proposal (10 particles)"];
(gold, ests) = z_estimates_comparison(
    [generate_occluded_bounce_tr()],
    get_returned_obs,
    obs_choicemap_to_vec_of_vec,
    specs;
    n_particles_when_producing_prev_traces=1,
    n_particles_goldstandard=10,
    n_estimates_per_spec=1
);
plot_z_estimate_comparison(gold, ests, labels)