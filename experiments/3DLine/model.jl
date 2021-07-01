using Gen
using Distributions
using Colors
using GLMakie
using StatsBase


# OK so all particles have inits of different latent states for moving_in_depth,
# but by the end of the filter run, they have switched to all moving in depth if this is true in tr.
# this doesn't appear to be possible. figure out why this happens tomorrow, but my guess right now
# is it has something to do with the fact that NaN particles are present, and these are ones where
# the proposal to moving in depth is wrong. in ess = -Inf, these keep going. in ess=nparticles, they dont.
# so either way at the end you end up with nan particles that proposed incorrectly on the first pass,
# and 

# commit test

include("../../src/ProbEstimates/ProbEstimates.jl")
ProbEstimates.use_perfect_weights!()
using .ProbEstimates: Cat, LCat

include("model_utils.jl")
include("model_hyperparams.jl")

neg_to_inf(x) = x <= 0 ? Inf : x
norm_3d(x, y, z) = sqrt(x^2 + y^2 + z^2)
round_to_pt1(x) = round(x, digits=1)

@gen (static) function initial_model()
    xₜ = { :xₜ } ~ Cat(unif(Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(unif(Ys()))
    heightₜ = { :heightₜ } ~ Cat(unif(Heights()))
    moving_in_depthₜ = { :moving_in_depthₜ } ~ bernoulli(.5)
    exact_r = norm_3d(xₜ, yₜ, heightₜ)
    rₜ = { :rₜ } ~ LCat(Rs())(discretized_gaussian(exact_r, 1.0, Rs()))
    exact_ϕ = { :exact_ϕ } ~ LCat(ϕs())(discretized_gaussian(asin(heightₜ / exact_r), .1, ϕs()))
    exact_θ = { :exact_θ } ~ LCat(θs())(discretized_gaussian(atan(yₜ / xₜ), .1, θs()))
    vₜ = { :vₜ } ~ LCat(Vels())(unif(Vels()))
    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ, exact_ϕ, exact_θ)
end

# x = back and forth
# y = left and right
# z = up and down (held constant in this model)
@gen (static) function step_model(moving_in_depthₜ₋₁, vₜ₋₁, heightₜ₋₁, xₜ₋₁, yₜ₋₁, rₜ₋₁, ephi, etheta)
    # TODO: experiment with 0.9 : 0.1 instead of 1.0 : 0.0
    moving_in_depthₜ = { :moving_in_depthₜ } ~ bernoulli(moving_in_depthₜ₋₁ ? 1.0 : 0.0)
    vₜ = { :vₜ } ~ LCat(Vels())(discretized_gaussian(vₜ₋₁, 0.2, Vels()))
    heightₜ = { :heightₜ } ~ Cat(moving_in_depthₜ ? onehot(heightₜ₋₁, Heights()) : discretized_gaussian(heightₜ₋₁ - vₜ, 1.0, Heights()))
    xₜ = { :xₜ } ~ Cat(moving_in_depthₜ ? discretized_gaussian(xₜ₋₁ + vₜ,  1.0, Xs()) : onehot(xₜ₋₁, Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(discretized_gaussian(yₜ₋₁ + vₜ, 1.0, Ys()))
    # Here: a stochastic mapping from (x, y, h) -> (r, θ, ϕ)
    # For now: just use dimension-wise discretized Gaussians.
    exact_r = norm_3d(xₜ, yₜ, heightₜ)
    rₜ = { :rₜ } ~ LCat(Rs())(discretized_gaussian(exact_r, 1.0, Rs()))
    exact_ϕ = { :exact_ϕ } ~ LCat(ϕs())(discretized_gaussian(asin(heightₜ / exact_r),
                                                             ϕstep(), ϕs()))
    exact_θ = { :exact_θ } ~ LCat(θs())(discretized_gaussian(atan(yₜ / xₜ), θstep(), θs()))
    return (moving_in_depthₜ, vₜ, heightₜ, xₜ, yₜ, rₜ, exact_ϕ, exact_θ)
end

# TODO: possible improvements:
# 1. maybe just make this a lookup table?
# 2. first round to low-res, then have cheap approximate lookup table
# @gen (static) function norm_3d(x, y, z)
#     xy = x^2 + y^2
#     xzy = xy + z^2
#     return sqrt(xzy)
# end

# θ is azimuth

# I think this might be redundant -- we are already producing obs in the step model. 
# seems like we should pass those exact variables here. also this suffers from the same problem as above
# if this is receiving a sample of r, then it could be shorter than x. 

@gen (static) function obs_model(moving_in_depth, v, height, x, y, r, exact_ϕ, exact_θ)
    # can't propose to these b/c they are the final observations we're scoring.
    # have to propose to the exact theta and phi.
    obs_ϕ = { :obs_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(exact_ϕ, 0.4, ϕs()))
    obs_θ = { :obs_θ } ~ LCat(θs())(truncated_discretized_gaussian(exact_θ, 0.4, θs()))
    return (obs_θ, obs_ϕ)
end

@gen (static) function step_proposal(
    moving_in_depthₜ₋₁, vₜ₋₁, heightₜ₋₁, xₜ₋₁, yₜ₋₁, rₜ₋₁, eϕ, eθ, θₜ, ϕₜ) # θ and ϕ are noisy
    # instead of sampling (x, y, h) then computing r (as we do in the model)
    # in the proposal we sample (r, x, y) and then compute h
    a = println(moving_in_depthₜ₋₁)
    moving_in_depthₜ = { :moving_in_depthₜ } ~ bernoulli(moving_in_depthₜ₋₁ ? 1.0 : 0.0)
    exact_θ = { :exact_θ } ~ LCat(θs())(truncated_discretized_gaussian(θₜ, 0.2, θs()))
    exact_ϕ = { :exact_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(ϕₜ, 0.2, ϕs()))
    rₜ = { :rₜ } ~ Cat(truncated_discretized_gaussian(rₜ₋₁, 3.0, Rs()))
    # now compute x, y, height (almost deterministically, plus some noise)
    exact_x = rₜ * cos(exact_ϕ) * cos(exact_θ)
    exact_y = rₜ * cos(exact_ϕ) * sin(exact_θ)
    exact_height = rₜ * sin(exact_ϕ)
    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant.
    xₜ = { :xₜ } ~ LCat(Xs())(
        moving_in_depthₜ ? truncated_discretized_gaussian(exact_x, 1.0, Xs()) : onehot(xₜ₋₁, Xs()))
    heightₜ = { :heightₜ } ~ LCat(Heights())(
        moving_in_depthₜ ? onehot(heightₜ₋₁, Heights()) : truncated_discretized_gaussian(
            exact_height, 1.0, Heights()))
    yₜ = { :yₜ } ~ LCat(Ys())(truncated_discretized_gaussian(exact_y, 1.0, Ys()))
    vₜ = { :vₜ } ~ LCat(Vels())(maybe_one_off(round_to_pt1(yₜ - yₜ₋₁), .4, Vels()))
end

@gen (static) function initial_proposal(θₜ, ϕₜ)
    moving_in_depthₜ = { :moving_in_depthₜ } ~ bernoulli(.5)
    exact_θ = { :exact_θ } ~ LCat(θs())(truncated_discretized_gaussian(θₜ, 0.2, θs()))
    exact_ϕ = { :exact_ϕ } ~ LCat(ϕs())(truncated_discretized_gaussian(ϕₜ, 0.2, ϕs()))
    # r_max on the first draw is guaranteed to not leave the cube
    r_max = minimum([Xs()[end] / (cos(exact_ϕ) * cos(exact_θ)),
                     neg_to_inf(Ys()[end] / (cos(exact_ϕ) * sin(exact_θ))), 
                     neg_to_inf(Ys()[1] / (cos(exact_ϕ) * sin(exact_θ))),
                     Heights()[end] / sin(exact_ϕ)])
    l = length(Rs())
    r_range = 1:r_max
    r_probvec = normalize(vcat(ones(Int64(r_range[end])), zeros(Int64(l-r_range[end]))))
    rₜ = { :rₜ } ~ LCat(Rs())(r_probvec)
    exact_x = rₜ * cos(exact_ϕ) * cos(exact_θ)
    exact_y = rₜ * cos(exact_ϕ) * sin(exact_θ)
    exact_height = rₜ * sin(exact_ϕ)
    # size in absolute terms is obtained by the az alt divs being discrete 
    # and az alt not having fixed xyz transforms when distant. 
    xₜ = { :xₜ } ~ LCat(Xs())(truncated_discretized_gaussian(exact_x, 1.0, Xs()))
    yₜ = { :yₜ } ~ LCat(Ys())(truncated_discretized_gaussian(exact_y, 1.0, Ys()))
    heightₜ = { :heightₜ } ~ LCat(Heights())(truncated_discretized_gaussian(exact_height, 1.0, Heights()))
    #    vₜ = { :vₜ } ~ LCat(Vels())(maybe_one_off(yₜ - yₜ₋₁, .4, Vels()))
    vₜ = { :vₜ } ~ LCat(Vels())(unif(Vels()))
end

function extract_submap_value(cmap, symlist) 
    lv = first(symlist)
    if lv == :val
        return cmap[:val]
    elseif length(symlist) == 1
        return cmap[lv]
    else
        cmap_sub = get_submap(cmap, lv)
        extract_submap_value(cmap_sub, symlist[2:end])
    end
end

    

function heatmap_pf_results(uw_traces, gt::Trace, nsteps)
    
    depth_indexer = [[:steps, i, :latents, :xₜ, :val] for i in 1:nsteps]
    height_indexer = [[:steps, i, :latents, :heightₜ, :val] for i in 1:nsteps]
                                  
    gray_cmap = range(colorant"white", stop=colorant"gray32", length=6)
    true_depth = [extract_submap_value(get_choices(gt), depth_indexer[i]) for i in 1:nsteps]
    true_height = [extract_submap_value(get_choices(gt), height_indexer[i]) for i in 1:nsteps]
    depth_matrix = zeros(nsteps, length(0:Xs()[end]) + 1)
    height_matrix = zeros(nsteps, length(0:Heights()[end]) + 1)
    for t in 1:nsteps
        for tr in uw_traces[end]
            depth_matrix[t, Int64(extract_submap_value(get_choices(tr), depth_indexer[t]))] += 1
            height_matrix[t, Int64(extract_submap_value(get_choices(tr), height_indexer[t]))] += 1
        end
    end
    # also plot the true x values
    fig = Figure(resolution=(1200,1200))
    ax_depth = fig[1, 1] = Axis(fig)
    hm_depth = heatmap!(ax_depth, depth_matrix, colormap=gray_cmap)    
    cbar = fig[1, 2] = Colorbar(fig, hm_depth, label="N Particles")
    ax_height = fig[2, 1] = Axis(fig)
    hm_height = heatmap!(ax_height, height_matrix, colormap=gray_cmap)
    cbar2 = fig[2, 2] = Colorbar(fig, hm_height, label="N Particles")
#    scatter!(ax, [o-.5 for o in observations], [t-.5 for t in 1:times], color=:skyblue2, marker=:rect, markersize=30.0)
    scatter!(ax_depth, [t for t in 1:nsteps], [tx for tx in true_depth], color=:orange, markersize=20.0)
    scatter!(ax_height, [t for t in 1:nsteps], [th for th in true_height], color=:orange, markersize=20.0)
    ax_depth.ylabel = "Depth"
    ax_height.ylabel = "Height"
    ax_height.xlabel = "Time"
    xlims!(ax_height, (.5, nsteps))
    xlims!(ax_depth, (.5, nsteps))
    ylims!(ax_height, (0.0, Heights()[end]+2))
    ylims!(ax_depth, (0.0, Xs()[end]+2))
    println("particle scores")
    println([get_score(tr) for tr in uw_traces[end]])
    display(fig)
    ax_moving_in_depth = fig[3, 1] = Axis(fig)
    hist!(ax_moving_in_depth,
          [extract_submap_value(
              get_choices(tr),
              [:steps, NSTEPS, :latents, :moving_in_depthₜ]) for tr in uw_traces[end]])
    return fig
end
