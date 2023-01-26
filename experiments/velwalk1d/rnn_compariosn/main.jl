#=
V1: 
Train on predicting y_t for a given σ.
RNN outputs a distribution over y_t.
Plot: x axis: new σ value used to generate test-time data
      y axis: error between true posterior predictive distribution over y_t
              and the RNN's/SMC's distribution over y_t
              (start with KL(true || approximate))
=#

includet("../model.jl")
includet("../inference.jl")
includet("../utils.jl")
using CairoMakie
using Memoization

# Parameters: VelStepStd(), ObsStd()
global obs_std = 1.0
function ObsStd()
    return obs_std
end

# rnn = get_trained_rnn()
function smc_predictive_posterior(smc_alg)
    return (params, obs_tr) -> smc_predictive_position_posterior(obs_tr, params, smc_alg)
end
params = (
    σs = [0.5, 1., 2., 3., 4., 6.],
    n_particles = 50,
    n_steps_per_run = 5,
    test_dataset_size = 200,
    default_σ = 1.0,
    # n_iters, print_every, plot_every
    training_params = (Int(400000/8), 1000, 100),
    inference_alg_labels = Any["approx proposal SMC", "smc_from_prior"], # "RNN"],
    inference_algs = Any[smc_predictive_posterior(smc_approx_proposal), smc_predictive_posterior(smc_from_prior)] #, rnn_predictive_posterior(rnn)]
)

include("rnn.jl")
(rnn, loss_record) = get_trained_rnn(params)

function rnn_predictive_position_posterior(rnn)
    return ((params, obs_tr) -> begin
        logprob_vector = reshape(runRNNOnObsTrace(rnn, obs_tr).tolist(), (:,))
        prob_vector = exp.(logprob_vector)
        @assert abs(sum(prob_vector) - 1.0) < 1e-5
        return prob_vector
    end)
end

push!(params.inference_alg_labels, "RNN")
push!(params.inference_algs, rnn_predictive_position_posterior(rnn))


function make_plot(params)
    test_datasets = [generate_test_dataset(σ, params) for σ in params.σs]
    mean_divergences_by_σ = [run_tests_on_dataset(dataset, params) for dataset in test_datasets]
    mean_divergences_by_alg = [[mean_divergences_by_σ[i][j] for i=1:length(params.σs)] for j in keys(params.inference_algs)]
    println("divergences: ", mean_divergences_by_σ)
    println("mean divergences by alg: ", mean_divergences_by_alg)

    # Plot
    f = Figure()
    ax = Axis(f[1, 1], xlabel="σ", ylabel="KL(inferred predictive posterior || true)", title="Inference performance (for σ=$(params.default_σ)) on data generated with different σs. NParticles=$(params.n_particles)")
    # lines!(ax, σs, rnn_mean_divergences, label="RNN")
    for (label, result) in zip(params.inference_alg_labels, mean_divergences_by_alg)
        lines!(ax, params.σs, result, label=label)
    end
    CairoMakie.axislegend(ax, valign=:top, halign=:left)
    return f
end

global test_datasets = Dict()
function generate_test_dataset(σ, params)
    global test_dataset
    if !haskey(test_datasets, σ)
        global obs_std = σ
        results = [simulate(model, (params.n_steps_per_run,)) for _=1:params.test_dataset_size]
        global obs_std = params.default_σ
        test_datasets[σ] = results
    elseif length(test_datasets[σ]) < params.test_dataset_size
        global obs_std = σ
        while length(test_datasets[σ]) < params.test_dataset_size
            push!(test_datasets[σ], simulate(model, (params.n_steps_per_run,)))
        end
        global obs_std = params.default_σ
    end
    return test_datasets[σ][1:params.test_dataset_size]
end

function run_tests_on_dataset(dataset, params)
    divergences = [[] for _ in params.inference_algs]
    for obs_tr in dataset
        exact_result = exact_predictive_position_posterior(obs_tr)
        for (i, inference_alg) in enumerate(params.inference_algs)
            # rnn_result = rnn_predictive_position_posterior(rnn, obs_tr, params)
            inferred_predictive_posterior = inference_alg(params, obs_tr)
            # if isinf(kl(exact_result, smc_result))
            #     println("Infinite divergence from $exact_result to $smc_result")
            # end
            push!(divergences[i], kl(inferred_predictive_posterior, exact_result))
            # push!(rnn_divergences, kl(exact_result, rnn_result))
        end
    end
    # println("SMC DIVS: $smc_divergences")
    # println("MEAN SMC DIVS: $([mean(divs) for divs in smc_divergences])")
    return [meannotnan(divs) for divs in divergences]
end
meannotnan(v) = mean(filter(!isnan, v))

# function run_tests_on_dataset(dataset, params)
#     divergences = [[] for _ in params.inference_algs]
#     for obs_tr in dataset
#         exact_result = exact_predictive_position_posterior(obs_tr)
#         for (i, inference_alg) in enumerate(params.inference_algs)
#             # rnn_result = rnn_predictive_position_posterior(rnn, obs_tr, params)
#             inference_result = smc_predictive_position_posterior(obs_tr, params, inference_alg)
#             # if isinf(kl(exact_result, smc_result))
#             #     println("Infinite divergence from $exact_result to $smc_result")
#             # end
#             push!(divergences[i], kl(inference_result, exact_result))
#             # push!(rnn_divergences, kl(exact_result, rnn_result))
#         end
#     end
#     # println("SMC DIVS: $smc_divergences")
#     # println("MEAN SMC DIVS: $([mean(divs) for divs in smc_divergences])")
#     return [mean(divs) for divs in divergences]
# end

@memoize function exact_predictive_position_posterior(obs_tr)
    println("Running exact inference")
    ProbEstimates.with_weight_type(:perfect, () -> begin
        exact_bayes_filter = enumeration_bayes_filter_from_groundtruth(
            obs_tr, initial_latent_model, step_latent_model, obs_model, (xₜ=Positions(),vₜ=Vels())
        ) |> DynamicModels.nest_all_addrs_at_val 
        latent_probability_grid = collect(exact_bayes_filter)[end]   
        obs_choicemaps = [choicemap((:obs => :val, y)) for y in Positions()]
        predictive_posterior = DynamicModels.latent_logdist_to_obs_dist(exact_bayes_filter, cm -> (cm[:xₜ => :val], cm[:vₜ => :val]), latent_probability_grid .- logsumexp(latent_probability_grid), obs_choicemaps)
        @assert sum(predictive_posterior) ≈ 1.0 "sum(predictive_posterior) = $(sum(predictive_posterior))"
        predictive_posterior
    end)
    # return [0. for _ in Positions()]
end

@memoize function xv_to_obsprob_vector(x, v)
    obs_choicemaps = [choicemap((:obs => :val, y)) for y in Positions()]
    obs_probs = [exp(generate(obs_model, (x, v), obs_choicemap)[2]) for obs_choicemap in obs_choicemaps]
    return obs_probs
end

function smc_predictive_position_posterior(obs_tr, params, smc_runner)
    (_, weighted_particles) = smc_runner(obs_tr, params.n_particles)
    x_dist = particles_to_x_dist(weighted_particles)
    y_dist = [0. for _ in Positions()]
    for (x, px) in enumerate(x_dist)
        obs_probs = xv_to_obsprob_vector(x, 0)
        y_dist += px .* obs_probs
    end
    return y_dist
end
function particles_to_x_dist(weighted_particles)
    dist = zeros(length(Positions()))

    trs = map(x -> x[1], weighted_particles[end])
    weights = map(x -> x[2], weighted_particles[end])
    nweights = exp.(weights .- logsumexp(weights))

    for (tr, w) in zip(trs, nweights)
        x = latents_choicemap(tr, get_args(tr)[1])[:xₜ => :val]
        dist[x] += w
    end
    return dist
end
# function rnn_predictive_position_posterior(rnn, obs_tr, params)
#     # TODO
# end