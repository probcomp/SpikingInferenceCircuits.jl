### Includes, ... ###
includet("../model/model.jl")
includet("../groundtruth_rendering.jl")
includet("../visualize.jl")
includet("../run_utils.jl")

include("../proposals/obs_aux_proposal.jl")
includet("../proposals/prior_proposal.jl")
includet("../proposals/nearly_locally_optimal_proposal.jl")
# TODO: includet("../proposals/bottom_up_proposal.jl")

ProbEstimates.use_perfect_weights!()

using DynamicModels: nest_at, obs_choicemap, latents_choicemap
put_at_end_of_pair(a, rest) = a => rest
put_at_end_of_pair(p::Pair, rest) = p.first => put_at_end_of_pair(p.second, rest)
extend_all_with(sym::Symbol, cm::ChoiceMap) = choicemap(
    ((put_at_end_of_pair(a, sym), v) for (a, v) in get_values_deep(cm))...
)
get_values_deep(cm) = Iterators.flatten((
    get_values_shallow(cm),
    (
        (a => r, v)
            for (a, sub) in get_submaps_shallow(cm)
                for (r, v) in get_values_deep(sub)
    )
))
@load_generated_functions()

categorical_from_matrix(matrix) =
    Tuple(keys(matrix)[categorical(reshape(matrix, length(matrix)))])

#=
Generate a trace where
1. at timestep 0, at (2, 5) with vel (2, 0); occ at 5
2. at timestep 1, at (4, 5) with vel (2, 0); occ at 5
3. at timestep 2, at (6, 5) with vel (2, 0); occ at 5 (so now it's behind the occluder)

Inference will be done over timestep 2.
=#
gt_tr, _ = generate(model, (2,), choicemap(
    (:init => :latents => :occₜ => :val, 5),
    (:init => :latents => :xₜ => :val, 2),
    (:init => :latents => :yₜ => :val, 5),
    (:init => :latents => :vxₜ => :val, 2),
    (:init => :latents => :vyₜ => :val, 0),
    
    (:steps => 1 => :latents => :occₜ => :val, 5),
    (:steps => 1 => :latents => :xₜ => :val, 4),
    (:steps => 1 => :latents => :yₜ => :val, 5),
    (:steps => 1 => :latents => :vxₜ => :val, 2),
    (:steps => 1 => :latents => :vyₜ => :val, 0),

    (:steps => 2 => :latents => :occₜ => :val, 5),
    (:steps => 2 => :latents => :xₜ => :val, 6),
    (:steps => 2 => :latents => :yₜ => :val, 5),
    (:steps => 2 => :latents => :vxₜ => :val, 2),
    (:steps => 2 => :latents => :vyₜ => :val, 0),
))

#=
Exact Bayesian inference, with occluder position fixed.
Enumerate every possible velocity setting for this step (which should deterministically restrict position).
For each one take the probability.  Get a 2D probability matrix for where the object is.
=#
"""
Returns a matrix probs where probs[x, y] is the posterior probability that
the new x and y position are x and y.
"""
function exact_inference_given_prevlatents_and_occluder(image_obs_choicemap::ChoiceMap, occₜ, latentsₜ₋₁)
    initial_tr, _ = generate(model, (0,), nest_at(:init => :latents, latentsₜ₋₁));
    logweights = [-Inf for _ in positions(SquareSideLength()), _ in positions(SquareSideLength())]
    for vxₜ in Vels(), vyₜ in Vels()
        xₜ = vxₜ + latentsₜ₋₁[:xₜ => :val]
        yₜ = vyₜ + latentsₜ₋₁[:yₜ => :val]
        @assert xₜ in Set(positions(SquareSideLength()))
        @assert yₜ in Set(positions(SquareSideLength()))
        newtr, weight, _, _ = update(
            initial_tr, (1,), (UnknownChange(),),
            merge(
                nest_at(:steps => 1 => :obs, image_obs_choicemap),
                nest_at(:steps => 1 => :latents, extend_all_with(:val, choicemap(
                    :xₜ => xₜ, :yₜ => yₜ, :vxₜ => vxₜ, :vyₜ => vyₜ, :occₜ => occₜ
                )))
            )
        );
        @assert logweights[xₜ, yₜ] == -Inf
        logweights[xₜ, yₜ] = weight
    end
    return exp.(logweights .- logsumexp(logweights))
end

function bottom_up_inference_given_occluder(image_obs_choicemap::ChoiceMap, occₜ)
    logweights = [-Inf for _ in positions(SquareSideLength()), _ in positions(SquareSideLength())]
    for xₜ in positions(SquareSideLength()), yₜ in positions(SquareSideLength())
        _, weight = generate(model, (0,), merge(
            nest_at(:init => :obs, image_obs_choicemap),
            nest_at(:init => :latents, extend_all_with(:val, choicemap(
                :xₜ => xₜ, :yₜ => yₜ, :vxₜ => 0, :vyₜ => 0, :occₜ => occₜ
            )))
        ))
        @assert logweights[xₜ, yₜ] == -Inf
        logweights[xₜ, yₜ] = weight
    end
    return exp.(logweights .- logsumexp(logweights))
end

exact_inference_results = exact_inference_given_prevlatents_and_occluder(
    obs_choicemap(gt_tr, 2), latents_choicemap(gt_tr, 2)[:occₜ => :val], latents_choicemap(gt_tr, 1)
)
bottom_up_inference_results = bottom_up_inference_given_occluder(
    obs_choicemap(gt_tr, 2), latents_choicemap(gt_tr, 2)[:occₜ => :val]
)

#=
Other proposals:
Set N = 10.
(1) Collect N samples from the locally-optimal proposal.  (That is, sample from the exact Bayesian infernece above.)
(2) Collect N samples from the bottom-up proposal.
=#
N = 10
locally_optimal_samples = [categorical_from_matrix(exact_inference_results) for _=1:N]
bottom_up_samples = [categorical_from_matrix(bottom_up_inference_results) for _=1:N]

samples_to_matrix(samples) = normalize([
    sum(s == (x, y) ? 1 : 0 for s in samples)
    for x in positions(SquareSideLength()), y in positions(SquareSideLength())
])

locally_optimal_matrix = samples_to_matrix(locally_optimal_samples)
bottom_up_matrix = samples_to_matrix(bottom_up_samples)

#=
Plotting.
Plot particle clouds from exact inference, and 2 particle approximations, and export the visuals.
=#
function plot_prob_matrix_squares!(ax, probability_matrix)
    maxprob = maximum(probability_matrix)
    to_alpha(prob) = 1.8 * prob # prob == 0 ? 0. : max(0.05, 1.8 * prob) # 0.8 * prob/maxprob
    sq = nothing
    for (idx, prob) in zip(keys(probability_matrix), probability_matrix)
        (x, y) = Tuple(idx)
        s = draw_sq!(ax, Observable(x), Observable(y), to_alpha(prob))
        if prob == maxprob
            sq = s
        end
    end
    return sq
end
function draw_vel_arrow!(ax, t, tr)
    cm = latents_choicemap(tr, t[])
    x, y, vx, vy = cm[:xₜ => :val], cm[:yₜ => :val], cm[:vxₜ => :val], cm[:vyₜ => :val]
    return arrows!(
        ax,
        [x - vx], [y - vy], [vx], [vy];
        color=colorant"seagreen", linewidth=4
    )
end
arrowshape() = PolyElement(color = :seagreen, points = Point2f0[
    (.16*x, .16*y + .2) for (x, y) in
    [(0, 1), (0, 2), (5, 2), (5, 3), (7, 1.5), (5, 0), (5, 1)]
])
function plot_obs_with_particle_dist!(figpos, tr, t, probability_matrix; title="")
    ax = Axis(figpos, aspect=DataAspect())
    hidedecorations!(ax)
    draw_obs!(ax, Observable(t), tr)
    sq = plot_prob_matrix_squares!(ax, probability_matrix)
    ar = draw_vel_arrow!(ax, Observable(t), tr)
    function make_legend(leg_figpos)
        l = Legend(leg_figpos, [sq, arrowshape()],
            ["Posterior over Position", "Ground Truth Motion into this Timestep"]; labelsize=30
        )
        l.tellheight = true; l.tellwidth = true
        return l
    end
    return make_legend
end
function plot_obs_with_particle_dist(tr, t, probability_matrix; title="")
    f = Figure()
    make_legend = plot_obs_with_particle_dist!(f[1, 1], tr, t, probability_matrix; title)
    make_legend(f[2, 1])

    return f
end
function plot_obs_particle_dists(gt_tr, t, titles, matrices)
    f = Figure(; resolution=(2000, 650))
    make_legend = nothing
    for (i, (title, matrix)) in enumerate(zip(titles, matrices))
        ml = plot_obs_with_particle_dist!(f[1, i], gt_tr, t, matrix; title)
        if isnothing(make_legend)
            make_legend = ml
        end
    end

    for (i, (l, caption)) in enumerate(zip(["d", "e", "f"], titles))
        Label(f.layout[1, i, Bottom()], "($l) $caption", textsize = 30 )
    end

    make_legend(f[2, :])
    return f
end

# f = plot_obs_with_particle_dist(gt_tr, 2, bottom_up_matrix)

f = plot_obs_particle_dists(gt_tr, 2,
    ["Exact Posterior", "$N Particles from Locally Exact Proposal", "$N Particles from Bottom-Up Proposal"],
    [exact_inference_results, locally_optimal_matrix, bottom_up_matrix]
)