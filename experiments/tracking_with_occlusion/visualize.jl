using GLMakie
using Colors
using FunctionalCollections

to_color(::Empty) = colorant"white"
to_color(::Object) = colorant"royalblue3"
to_color(::Occluder) = colorant"indianred"

to_idx(::Empty) = 1
to_idx(::Object) = 2
to_idx(::Occluder) = 3
to_color(i::Int) = to_color(PixelColors()[i])

### Drawing utils ###

function hollow_rect!(ax, pts; color)
    # pts = (x1, y1, x2, y2)
    linesegments!(
        ax,
        lift(((x1, y1, x2, y2),) -> [
            (x1, y1), (x1, y2),
            (x1, y2), (x2, y2),
            (x2, y2), (x2, y1),
            (x2, y1), (x1, y1)
        ], pts);
        color, linewidth=5
    )
end

### Trace data utils ###

to_color_matrix(vec_of_vecs_of_pixel_color) = [
	to_idx(vec_of_vecs_of_pixel_color[x][y])
		for x=1:length(vec_of_vecs_of_pixel_color),
			y=1:length(vec_of_vecs_of_pixel_color[1])	
]

observed_imgs(tr) = [
    tr[:init => :obs][1],
    (
        tr[:steps => t => :obs][1]
        for t=1:(get_args(tr)[1])
    )...
]

### Figure making / trace drawing ###

sq_left(x) = x - 0.4
sq_bot(y) = y - 0.4
occ_left(occ) = occ - 0.4
sq_left(tr, t)  = sq_left(latents_choicemap(tr, t)[:xₜ => :val])
sq_bot(tr, t)   = sq_bot(latents_choicemap(tr, t)[:yₜ => :val])
sq_top(args...)   = sq_bot(args...)  + SquareSideLength() - 0.2
sq_right(args...) = sq_left(args...) + SquareSideLength() - 0.2
occ_left(tr, t)     = occ_left(latents_choicemap(tr, t)[:occₜ => :val])
occ_right(args...)    = occ_left(args...) + OccluderLength() - 0.2

sq_x_center(tr, t) = latents_choicemap(tr, t)[:xₜ => :val]
sq_y_center(tr, t) = latents_choicemap(tr, t)[:yₜ => :val]
sq_center(tr, t) = Point2(sq_x_center(tr, t), sq_y_center(tr, t))

function draw_obs!(ax, t, tr)
    obs = observed_imgs(tr)
    heatmap!(ax, @lift(to_color_matrix(obs[$t + 1])), colormap=map(to_color, PixelColors()))
end


function draw_obs!(ax, t, obs::Vector{FunctionalCollections.PersistentVector{FunctionalCollections.PersistentVector{Any}}})
    heatmap!(ax, @lift(to_color_matrix(obs[$t + 1])), colormap=map(to_color, PixelColors()))
end

function draw_gt_sq!(ax, t, tr)
    # hollow_rect!(
    #     ax,
    #     lift(t -> (sq_left(tr, t), sq_bot(tr, t), sq_right(tr, t), sq_top(tr, t)), t),
    #     color=colorant"seagreen"
    # )
    scatter!(ax, lift(t -> [sq_center(tr, t)], t), color=colorant"seagreen", markersize=30)
end

function draw_gt_occ!(ax, t, tr)
    hollow_rect!(
        ax,
        lift(t -> (occ_left(tr, t), 0.6, occ_right(tr, t), ImageSideLength() + 0.4), t),
        color=colorant"seagreen"
    )
end

function draw_tr(tr)
    fig = Figure()
    ax = Axis(fig[1, 1])
    t = Observable(0)
    draw_obs!(ax, t, tr)
    draw_gt_sq!(ax, t, tr)
    draw_gt_occ!(ax, t, tr)
    return (fig, t)
end

function draw_obs(tr)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="Observed Image")
    t = Observable(0)
    draw_obs!(ax, t, tr)
    return (fig, t)
end

function draw_obs(obs::Vector{FunctionalCollections.PersistentVector{FunctionalCollections.PersistentVector{Any}}})
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="ANN Predicted Image")
    t = Observable(0)
    draw_obs!(ax, t, obs)
    return (fig, t)
end




### Inference drawing ###
function draw_sq!(ax, x, y, alpha)
    poly!(
        ax,
        @lift(Rect(sq_left($x) + 0.1, sq_left($y) + 0.1, SquareSideLength() - 0.4, SquareSideLength() - 0.4)),
        color=RGBA(0, 0, 0, alpha)
    )
end
function draw_particle_sq!(ax, t, tr, num_particles) # tr = observable giving trace at time $t
    draw_sq!(ax,
        @lift(latents_choicemap($tr, $t)[:xₜ => :val]),
        @lift(latents_choicemap($tr, $t)[:yₜ => :val]),
        min(1., 2.0/num_particles)
    )
    # poly!(
    #     ax,
    #     @lift(Rect(sq_left($tr, $t) + 0.1, sq_bot($tr, $t) + 0.1, SquareSideLength() - 0.4, SquareSideLength() - 0.4)),
    #     color=RGBA(0, 0, 0, min(1., 2.0/num_particles))
    # )
end
function draw_particle_occ_gt!(ax, t, tr, num_particles)
    poly!(
        ax,
        @lift(Rect(occ_left($tr, $t), 1, OccluderLength() - 0.2, 1)),
        color=RGBA(0, 0, 0, min(0.95, 2.0/num_particles))
    )
end
function draw_particles!(drawfn, ax, t, trs_indexed_by_time)
    isempty(trs_indexed_by_time) && return;
    for i=1:length(trs_indexed_by_time[1])
        drawfn(ax, t, @lift(trs_indexed_by_time[$t + 1][i]), length(trs_indexed_by_time[1]))
    end
end

function draw_vel!(ax, t, tr, num_particles)
    vx(tr, t) = time_to_vel(tr)(t)[1]
    vy(tr, t) = time_to_vel(tr)(t)[2]
    poly!(
        ax,
        @lift(Rect(vx($tr, $t) - 0.4, vy($tr, $t) - 0.4, 0.8, 0.8)),
        color=RGBA(0, 0, 0, min(0.95, 2.0/num_particles))
    )
end
plot_point!(ax, t, time_to_pt, domain;
    n_backtrack=0, color, markersize=30, marker=:circle
) = scatter!(ax,
        @lift([time_to_pt($t)[1]]),
        @lift([time_to_pt($t)[2]]);
        color, markersize, marker
    )
time_to_vel(tr) = t -> (latents_choicemap(tr, t)[:vxₜ => :val], latents_choicemap(tr, t)[:vyₜ => :val])


# particles here is unweighted_trs
function draw_gt_and_particles(tr, particles, inference_method_str)
    fig = Figure(resolution=(800, 1500))
    t = Observable(0)

    vel_ax = Axis(fig[1, 1], aspect=DataAspect(), title="Velocity")
    draw_particles!(draw_vel!, vel_ax, t, particles)
    pts = plot_point!(vel_ax, t, time_to_vel(tr), Vels(); n_backtrack=2, color=colorant"seagreen")
    xlims!(vel_ax, (first(Vels()) - 0.5, last(Vels()) + 0.5))
    ylims!(vel_ax, (first(Vels()) - 0.5, last(Vels()) + 0.5))
    vel_ax.xticks=Vels()
    vel_ax.yticks=Vels()

    l1 = GridLayout()
    fig[2:3, 1] = l1
    l1[1, 1] = ax2d = Axis(fig[2, 1], aspect=DataAspect(), title="2D world")
    obs = draw_obs!(ax2d, t, tr)
    inf = draw_particles!(draw_particle_sq!, ax2d, t, particles)
    gt = draw_gt_sq!(ax2d, t, tr)
    draw_gt_occ!(ax2d, t, tr)
    xlims!(ax2d, (0.5, last(Positions()) + 0.5))

    l1[2, 1] = ax1d = Axis(fig[3, 1], aspect=DataAspect(), title="Inferred occluder position")
    hideydecorations!(ax1d)
    draw_particles!(draw_particle_occ_gt!, ax1d, t, particles)
    linkxaxes!(ax2d, ax1d)

    rowsize!(l1, 1, Relative(  (last(Positions()) - 1) / last(Positions())     ))
    rowsize!(l1, 2, Relative(1/last(Positions())))
    # ax2d.height = Relative(20/21)
    # ax1d.height = Relative(1/21)
    # ax2d.tellheight = true
    # ax1d.tellheight = true
    # trim!(fig.layout)

    hyperparam_label = Label(fig, "p_pixel_flip = $(ColorFlipProb()). OccOneOffProb = $(OccOneOffProb()).  VelStd = $(VelStd()).")
    inference_label = Label(fig, inference_method_str)
    hyperparam_label.tellwidth = false; inference_label.tellwidth = false
    fig[4, :] = hyperparam_label; fig[5, :] = inference_label;

    leg = Legend(fig[6, :],
        [
            PolyElement(color=:indianred),
            PolyElement(color=:royalblue3, points = Point2[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]),
            [LineElement(color=:seagreen), MarkerElement(marker=:circle, color=:seagreen, markersize=10)],
            [
                PolyElement(color=:gray),
                PolyElement(color=:black, points = Point2[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
            ]
        ],
        ["Observed image: occluder", "Observed image: ball", "Ground-Truth", "Inferred Positions"]
    )
    leg.orientation = :vertical


    return (fig, t)
end
function draw_gt_particles_img_only(tr, particles, inference_method_str)
    fig = Figure(resolution=(1000, 1000))
    t = Observable(0)
    ax2d = Axis(fig[1, 1], aspect=DataAspect(), title="2D world")
    obs = draw_obs!(ax2d, t, tr)
    inf = draw_particles!(draw_particle_sq!, ax2d, t, particles)
    gt = draw_gt_sq!(ax2d, t, tr)
    draw_gt_occ!(ax2d, t, tr)
    xlims!(ax2d, (0.5, last(Positions()) + 0.5))
    colsize!(fig.layout, 1, Relative(1))

    hyperparam_label = Label(fig, "p_pixel_flip = $(ColorFlipProb()). OccOneOffProb = $(OccOneOffProb()).  VelStd = $(VelStd()).")
    inference_label = Label(fig, inference_method_str)
    hyperparam_label.tellwidth = false; inference_label.tellwidth = false
    hyperparam_label.tellheight = true; inference_label.tellheight = true
    fig[2, 1] = hyperparam_label; fig[3, 1] = inference_label;

    leg = Legend(fig[4, 1],
        [
            PolyElement(color=:indianred),
            PolyElement(color=:royalblue3, points = Point2[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]),
            [LineElement(color=:seagreen), MarkerElement(marker=:circle, color=:seagreen, markersize=10)],
            [
                PolyElement(color=:gray),
                PolyElement(color=:black, points = Point2[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
            ]
        ],
        ["Observed image: occluder", "Observed image: ball", "Ground-Truth", "Inferred Positions"]
    )
    leg.orientation = :vertical
    leg.tellheight = true

    trim!(fig.layout)

    return (fig, t)
end

### Animation ###
FRAMERATE() = 1

function animate(t, T)
    for _t = 0:T
        t[] = _t
        sleep(1/FRAMERATE())
    end
end

make_video(fig, t, T, filename) =
    record(fig, filename, 0:T; framerate=FRAMERATE()) do _t
        t[] = _t
    end
