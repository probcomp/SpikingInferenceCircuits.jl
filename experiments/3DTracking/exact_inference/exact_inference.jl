using GenDiscreteHMM
using Serialization

hmm = GenDiscreteHMM.deserialize_hmm_from_gfs(
    "medium_hmm_contents.jld",
    [
        choicemap(
            (:vxₜ, vx), (:vyₜ, vy), (:vzₜ, vz),
            (:xₜ, x), (:yₜ, y), (:zₜ, z)
        )
        for vx in Vels() for vy in Vels() for vz in Vels() for x in Xs() for y in Ys() for z in Zs()
    ],
    [
        choicemap(
            (:true_ϕ, ϕ), (:true_θ, θ), (:rₜ, r)
        )
        for ϕ in ϕs() for θ in θs() for r in Rs()
    ],
    # generative functions
    obs_model;
    likelihood_arg_format=(
        (u, _) -> (u[:true_ϕ], u[:true_θ])
    ),
    y_depends_only_on_u = true
);

function get_azalt_for_xyz(x, y, z)
    tr, _ = generate(transient_state_model, (x, y, z))
    return (tr[:true_ϕ], tr[:true_θ])
end

# observation_sequence = [
#     get_azalt_for_xyz(3, 2, z) for z=1:8
# ]
observation_sequence = [
    (i, 0.) for i=0.0:0.1:1.4
]

# Construct a filter that outputs the posterior over 
# assignments to (true_ϕ, true_θ)
partition = [
    # [choicemap((:true_ϕ, ϕ), (:true_θ, θ), (:rₜ, r)) for r in Rs()]
    [choicemap((:true_ϕ, ϕ), (:true_θ, θ), (:rₜ, r)) for r in Rs()]
    for ϕ in ϕs() for θ in θs()
]
filter = LabeledDiscreteExactBayesFilter(
    hmm,
    [choicemap((:obs_ϕ, ϕ), (:obs_θ, θ)) for (ϕ, θ) in observation_sequence],
    partition,
    only_query_u=true
);
@time results = collect(filter);

function result_to_matrix(pvec)
    matrix = [-1.0 for _ in ϕs(), _ in θs()]
    ordering = [(ϕ, θ) for ϕ in ϕs() for θ in θs()]
    for (i, (ϕ, θ)) in enumerate(ordering)
        matrix[Int(ϕ*10 + 1), Int(θ * 10 + 15)] = pvec[i]
    end
    return matrix
end

# probability matrix over the true azimuth/altitude
# at each timestep, from exact bayes filter
az_alt_matrices = [
    result_to_matrix(result)
    for result in results
]