### Visualize transitions represented by the datapoints ###
#=
Each datapoint is associated with
1. A transition xₜ₋₁ → yₜ
2. A "gold standard" weighted particle collection {xₜ}
3. A "estimate" weighted particle collection {xₜ}

I would like to make a figure which visualizes:
(1) xₜ₋₁
(2) yₜ
(3) the "gold standard" particles {xₜ} overlayed on yₜ
(4) the "estimate standard" particles {xₜ} overlayed on yₜ
=#

# we don't need to pass in (xₜ₋₁, yₜ) since we can get
# these from the extended traces
function visualize_datapoint(gold_standard, estimate)
    golds = [tr for (tr, _) in gold_standard]
    ests = [tr for (tr, _) in estimate]

    # get a trace extended with the observation
    an_extended_tr = first(golds)
    _t = get_args(an_extended_tr)[1]
    # println("t = $_t")
    t = Observable(_t)
    # TODO: I probably need to turn `t` into an observable
    
    f = Figure()
    ax1 = Axis(f[1, 1], aspect=DataAspect(), title="Particles from gold-standard estimate")
    yₜ_viz = draw_obs!(ax1, t, an_extended_tr)
    xₜ₋₁_pos_viz = draw_gt_sq!(ax1, lift(t -> t - 1, t), an_extended_tr)
    xₜ₋₁_vel_viz = draw_vel_arrow!(ax1, lift(t -> t - 1, t), an_extended_tr)
    gold_viz = draw_particle_squares!(ax1, t, golds)

    ax2 = Axis(f[1, 2], aspect=DataAspect(), title="Particles from z estimate")
    yₜ_viz = draw_obs!(ax2, t, an_extended_tr)
    xₜ₋₁_pos_viz = draw_gt_sq!(ax2, lift(t -> t - 1, t), an_extended_tr)
    xₜ₋₁_vel_viz = draw_vel_arrow!(ax2, lift(t -> t - 1, t), an_extended_tr)
    est_viz = draw_particle_squares!(ax2, t, ests)

    Legend(
        f[2, :],
        [
            [
                PolyElement(color=:indianred),
                PolyElement(
                    color=:royalblue3, 
                    points = Point2f0[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
                )
            ],
            xₜ₋₁_pos_viz, xₜ₋₁_vel_viz,
            [
                PolyElement(color=:gray),
                PolyElement(color=:black, points = Point2f0[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
            ]
        ],
        [
            "yₜ",
            "xₜ₋₁ position", "xₜ₋₁ velocity",
            "Proposed particles [alpha = number of particles proposed there, ignoring weights]"
        ]
    )
    trim!(f.layout)

    return f
end
function draw_particle_squares!(ax, t, trs)
    isempty(trs) && return;
    for tr in trs
        draw_particle_sq!(ax, t, Observable(tr), length(trs))
    end
end
function draw_vel_arrow!(ax, t, tr)
    cm = latents_choicemap(tr, t[])
    x, y, vx, vy = cm[:xₜ => :val], cm[:yₜ => :val], cm[:vxₜ => :val], cm[:vyₜ => :val]
    arrows!(
        ax,
        [x - vx], [y - vy], [vx], [vy];
        color=colorant"seagreen"
    )
end

### Get datapoint closest to a given (x, y) position ###
function find_closest(golds, ests, (x, y))
    # first find the closest gold;
    # then find the closest est for that gold
    # currently for -Inf just return the first one

    if isinf(x)
        closest_gold_idx = findfirst([isinf(z) for (z, _) in golds])
    else
        closest_gold_idx = argmin([isnan(z) ? Inf : abs(z - x) for (z, _) in golds])
    end

    if isinf(y)
        closest_est_idx = findfirst([isinf(z) for (z, _) in ests[closest_gold_idx]])
    else
        closest_est_idx = argmin([isnan(z) ? Inf : abs(z - y) for (z, _) in ests[closest_gold_idx]])
    end

    println("$closest_gold_idx | $closest_est_idx")

    (closest_gold_z, gold_proposals) = golds[closest_gold_idx]
    (closest_est_z, est_proposals) = ests[closest_gold_idx][closest_est_idx]
    return (closest_gold_z, gold_proposals, closest_est_z, est_proposals)
end

#=
i = index of the plot we are looking at
=#
function visualize_closest_datapoint(i, golds, ests, (x, y))
    (closest_gold_z, gold_proposals, closest_est_z, est_proposals) = find_closest(golds, ests[i], (x, y))
    visualize_datapoint(gold_proposals, est_proposals)
end