using Memoization
using Gen

### Includes, ... ###
includet("../model/model_noflipvars.jl")
includet("../groundtruth_rendering.jl")
includet("../visualize.jl")
includet("../run_utils.jl")

includet("../proposals/obs_aux_proposal.jl")
includet("../proposals/prior_proposal.jl")
includet("../proposals/nearly_locally_optimal_proposal.jl")
# TODO: includet("../proposals/bottom_up_proposal.jl")

ProbEstimates.use_perfect_weights!()

set_use_aux_vars!(false)

using DynamicModels: obs_choicemap, latents_choicemap
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
Exact Bayesian inference, with occluder position fixed.
Enumerate every possible velocity setting for this step (which should deterministically restrict position).
For each one take the probability.  Get a 2D probability matrix for where the object is.
=#
"""
Returns a matrix probs where probs[x, y] is the posterior probability that
the new x and y position are x and y.
"""
function exact_inference_given_prevlatents_and_occluder(image_obs_choicemap::ChoiceMap, occₜ, latentsₜ₋₁)
    initial_tr, _ = generate(model_noflipvars, (0,), DynamicModels.nest_at(:init => :latents, latentsₜ₋₁));
    logweights = [-Inf for _ in positions(SquareSideLength()), _ in positions(SquareSideLength())]
    for vxₜ in Vels(), vyₜ in Vels()
        xₜ = vxₜ + latentsₜ₋₁[:xₜ => :val]
        yₜ = vyₜ + latentsₜ₋₁[:yₜ => :val]
        @assert xₜ in Set(positions(SquareSideLength()))
        @assert yₜ in Set(positions(SquareSideLength()))
        newtr, weight, _, _ = update(
            initial_tr, (1,), (UnknownChange(),),
            merge(
                DynamicModels.nest_at(:steps => 1 => :obs, image_obs_choicemap),
                DynamicModels.nest_at(:steps => 1 => :latents, extend_all_with(:val, choicemap(
                    :xₜ => xₜ, :yₜ => yₜ, :vxₜ => vxₜ, :vyₜ => vyₜ, :occₜ => occₜ
                )))
            )
        );
        @assert logweights[xₜ, yₜ] == -Inf
        logweights[xₜ, yₜ] = weight
    end
    return exp.(logweights .- logsumexp(logweights))
end

function model_down_inference_given_prevlatents(occₜ, latentsₜ₋₁)
    initial_tr, _ = generate(model_noflipvars, (0,), DynamicModels.nest_at(:init => :latents, latentsₜ₋₁));
    logweights = [-Inf for _ in positions(SquareSideLength()), _ in positions(SquareSideLength())]
    for vxₜ in Vels(), vyₜ in Vels()
        xₜ = vxₜ + latentsₜ₋₁[:xₜ => :val]
        yₜ = vyₜ + latentsₜ₋₁[:yₜ => :val]
        @assert xₜ in Set(positions(SquareSideLength()))
        @assert yₜ in Set(positions(SquareSideLength()))
        newtr, weight, _, _ = update(
            initial_tr, (1,), (UnknownChange(),),
            DynamicModels.nest_at(:steps => 1 => :latents, extend_all_with(:val, choicemap(
                :xₜ => xₜ, :yₜ => yₜ, :vxₜ => vxₜ, :vyₜ => vyₜ, :occₜ => occₜ
            )))
        );
        @assert logweights[xₜ, yₜ] == -Inf
        logweights[xₜ, yₜ] = weight
    end
    return exp.(logweights .- logsumexp(logweights))
end

function bottom_up_inference_given_occluder(image_obs_choicemap::ChoiceMap, occₜ)
    logweights = [-Inf for _ in positions(SquareSideLength()), _ in positions(SquareSideLength())]
    for xₜ in positions(SquareSideLength()), yₜ in positions(SquareSideLength())
        _, weight = generate(model_noflipvars, (0,), merge(
            DynamicModels.nest_at(:init => :obs, image_obs_choicemap),
            DynamicModels.nest_at(:init => :latents, extend_all_with(:val, choicemap(
                :xₜ => xₜ, :yₜ => yₜ, :vxₜ => 0, :vyₜ => 0, :occₜ => occₜ
            )))
        ))
        @assert logweights[xₜ, yₜ] == -Inf
        logweights[xₜ, yₜ] = weight
    end
    return exp.(logweights .- logsumexp(logweights))
end

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
function draw_predicted_vel_arrow!(ax, t, tr)
    cm = latents_choicemap(tr, t[] - 1)
    x, y, vx, vy = cm[:xₜ => :val], cm[:yₜ => :val], cm[:vxₜ => :val], cm[:vyₜ => :val]
    # println("x, y, vx, vy  = $((x, y, vx, vy ))")
    return arrows!(
        ax,
        [x], [y], [vx], [vy];
        color=colorant"seagreen", linewidth=4
    )
end
arrowshape() = PolyElement(color = :seagreen, points = Point2[
    (.16*x, .16*y + .2) for (x, y) in
    [(0, 1), (0, 2), (5, 2), (5, 3), (7, 1.5), (5, 0), (5, 1)]
])
function plot_scenario!(figpos, tr; title="", do_predicted_vel=false)
    t = get_args(tr)[1]
    ax = Axis(figpos, aspect=DataAspect(); title)
    hidedecorations!(ax)
    draw_obs!(ax, Observable(t), tr)
    draw_arrow! = do_predicted_vel ? draw_predicted_vel_arrow! : draw_vel_arrow!
    ar = draw_arrow!(ax, Observable(t), tr)
    return ax
end
function plot_obs_with_particle_dist!(figpos, tr, t, probability_matrix; title="")
    ax = Axis(figpos, aspect=DataAspect(); title)
    hidedecorations!(ax)
    draw_obs!(ax, Observable(t), tr)
    sq = plot_prob_matrix_squares!(ax, probability_matrix)
    ar = draw_vel_arrow!(ax, Observable(t), tr)
    function make_legend(leg_figpos)
        l = Legend(leg_figpos, [sq, arrowshape()],
            ["Posterior over Position", "Ground Truth Motion into this Timestep"]
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
    f = Figure(; resolution=(2000, 450))
    make_legend = nothing
    for (i, (title, matrix)) in enumerate(zip(titles, matrices))
        ml = plot_obs_with_particle_dist!(f[1, i], gt_tr, t, matrix; title)
        if isnothing(make_legend)
            make_legend = ml
        end
    end
    make_legend(f[2, :])
    return f
end

### Inference results ###

get_exact_inference_results(gt_tr) = exact_inference_given_prevlatents_and_occluder(
    obs_choicemap(gt_tr, 2), latents_choicemap(gt_tr, 2)[:occₜ => :val], latents_choicemap(gt_tr, 1)
)
get_bottom_up_distribution(gt_tr) = bottom_up_inference_given_occluder(
    obs_choicemap(gt_tr, 2), latents_choicemap(gt_tr, 2)[:occₜ => :val]
)
get_top_down_distribution(gt_tr) = model_down_inference_given_prevlatents(latents_choicemap(gt_tr, 2)[:occₜ => :val], latents_choicemap(gt_tr, 1))
get_hybrid_distribution(gt_tr) = 1/2 * get_bottom_up_distribution(gt_tr) + 1/2 * get_top_down_distribution(gt_tr)

function get_is_sample(n_particles, exact_inference_dist, proposal_dist)
    samples = [categorical_from_matrix(proposal_dist) for _=1:n_particles]
    weights = [exact_inference_dist[x, y]/proposal_dist[x, y] for (x, y) in samples]
    weights = [isnan(w) ? 0. : w for w in weights]
    if !(sum(weights) > 0.)
        # println("weights = $weights")
        return samples[uniform_discrete(1, length(samples))]
    end
    probs = normalize(weights)
    a = categorical(probs)
    return samples[a]
end
function get_raw_sample(n_particles, exact_inference_dist, proposal_dist)
    return categorical_from_matrix(proposal_dist)
end
function get_5th_mean_95th(get_sample, exact_inference_results, proposal, n_particles; n_runs=10_000)
    cnt = 0
    vals = []
    for _=1:n_runs
        (x, y) = get_sample(n_particles, exact_inference_results, proposal)
        res = exact_inference_results[x, y]
        if !isnan(res)
            push!(vals, res)
        end
        cnt += 1
    end

    mean = sum(vals) / cnt
    fifth = get_fractile(vals, 0.35)
    ninetyfifth = get_fractile(vals, 0.65)
    return (fifth, mean, ninetyfifth)
end
function get_fractile(vals, frac)
    sorted = sort(vals)
    return sorted[Int(round(frac * length(sorted)))]
end

### Quant Plots ###
function plot_quant_comparison!(ax, gt_tr)
    exact_inference_results = get_exact_inference_results(gt_tr)
    bottom_up_distribution = get_bottom_up_distribution(gt_tr)
    top_down_distribution = get_top_down_distribution(gt_tr)
    hybrid_distribution = get_hybrid_distribution(gt_tr)

    ns = 1:20
    specs = [
        ("Exact posterior", get_raw_sample, exact_inference_results, :black, :solid),
        ("Data-Driven Proposal Distribution", get_raw_sample, bottom_up_distribution, :blue, :dash),
        ("Model-Driven Proposal Distribution", get_raw_sample, top_down_distribution, :green, :dash),
        ("Data-Driven Proposal\n+ Model-Based Scoring", get_is_sample, bottom_up_distribution, :blue, :solid),
        ("Model-Driven Proposal Distribution\n+ Model-Based Scoring", get_is_sample, top_down_distribution, :green, :solid),
        ("Hybrid Proposal\n+ Model-Based Scoring", get_is_sample, hybrid_distribution, :red, :solid),
    ]
    plots = []
    labels = []
    trips = []
    for (label, get_sample, proposal, color, linestyle) in specs
        try
            triples = [get_5th_mean_95th(get_sample, exact_inference_results, proposal, n) for n in ns]
            fifths = map(x -> x[1], triples)
            means = map(x -> x[2], triples)
            ninetyfifths = map(x -> x[3], triples)
            ln = lines!(ax, ns, means; label, color, linestyle)
            # bnd = band!(ax, ns, fifths, ninetyfifths)
            push!(plots, (ln, nothing))
            push!(labels, label)
            push!(trips, triples)
        catch e
            println("label = $label")
            throw(e)
        end
    end
    # axislegend(ax, position=:rc)
    return (plots, labels, trips)
end

function make_plot(gt_tr)
    f = Figure()
    ax = Axis(f[1, 1], ylabel="E[P(xₜ | xₜ₋₁, yₜ)] of generated sample", xlabel="Number of particles used for inference"; textsize=26)
    plot_quant_comparison!(ax, gt_tr)
    f
end

function make_plot_with_visual(gt_tr)
    f = Figure()
    plot_scenario!(f[1, 1], gt_tr)
    quant_ax = Axis(f[2, 1], ylabel="E[P(xₜ | xₜ₋₁, yₜ)] of generated sample", xlabel="Number of particles used for inference", textsize=26)
    plot_quant_comparison!(quant_ax, gt_tr)
    f
end

function make_plots_with_visuals(title_tr_pairs)
    f = Figure(; resolution=(2000, 600))#(100 + 400 * length(title_tr_pairs), 800))
    plots, labels = nothing, nothing
    trips = []
    for (i, (title, gt_tr)) in enumerate(title_tr_pairs)
        plot_scenario!(f[1, i], gt_tr; title, do_predicted_vel=true)
        quant_ax = Axis(f[2, i], ylabel="E[P(xₜ | xₜ₋₁, yₜ)] of generated sample", xlabel="Number of particles used for inference", textsize=26)
        plots, labels, trips_ = plot_quant_comparison!(quant_ax, gt_tr)
        push!(trips, trips_)
    end

    leg_layout = f[:, length(title_tr_pairs) + 1] = GridLayout()
    leg_layout[1, 1] = l2 = Legend(leg_layout[1, 2],
        [arrowshape()],
        ["MAP Motion Prediction"]
    )
    leg_layout[2, 1] = l1 = Legend(leg_layout[1, 1], map(x -> x[1], plots), convert(Vector{String}, labels))
    l1.tellheight = true; l2.tellheight = true

    leg_layout.tellheight=true
    rowsize!(f.layout, 1, Relative(0.4))
    (f, trips)
end