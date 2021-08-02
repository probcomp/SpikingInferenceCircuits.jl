using GLMakie
using Colors

to_color(::Empty) = colorant"white"
to_color(::Object) = colorant"gold"
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
        color
    )
end

### Trace data utils ###

to_color_matrix(vec_of_vecs_of_pixel_color) = [
	to_idx(vec_of_vecs_of_pixel_color[x][y])
		for x=1:length(vec_of_vecs_of_pixel_color),
			y=1:length(vec_of_vecs_of_pixel_color[1])	
] |> transpose

observed_imgs(tr) = [
    tr[:init => :obs][1],
    (
        tr[:steps => t => :obs][1]
        for t=1:(get_args(tr)[1])
    )...
]

### Figure making / trace drawing ###

sq_left(tr, t)  = latents_choicemap(tr, t)[:xₜ => :val] - 0.5
sq_bot(tr, t)   = latents_choicemap(tr, t)[:yₜ => :val] - 0.5
sq_top(tr, t)   = sq_bot(tr, t)  + SquareSideLength() 
sq_right(tr, t) = sq_left(tr, t) + SquareSideLength()
occ_left(tr, t)     = latents_choicemap(tr, t)[:occₜ => :val] - 0.4
occ_right(tr, t)    = occ_left(tr, t) + OccluderLength() - 0.2

function draw_obs!(ax, t, tr)
    obs = observed_imgs(tr)
    heatmap!(ax, @lift(to_color_matrix(obs[$t + 1])), colormap=map(to_color, PixelColors()))
end
function draw_gt_sq!(ax, t, tr)
    hollow_rect!(
        ax,
        lift(t -> (sq_left(tr, t), sq_bot(tr, t), sq_right(tr, t), sq_top(tr, t)), t),
        color=colorant"seagreen"
    )
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

### Inference drawing ###
function draw_particle_sq_gt!(ax, t, tr) # tr = observable giving trace at time $t
    poly!(
        ax,
        @lift(Rect(sq_left($tr, $t), sq_bot($tr, $t), SquareSideLength(), SquareSideLength())),
        color=RGBA(0, 0, 0, 0.2)
    )
end
function draw_particle_occ_gt!(ax, t, tr)
    poly!(
        ax,
        @lift(Rect(occ_left($tr, $t), 1, OccluderLength(), 1)),
        color=RGBA(0, 0, 0, 0.2)
    )
end
function draw_particles_gt!(drawfn, ax, t, trs_indexed_by_time)
    isempty(trs_indexed_by_time) && return;
    for i=1:length(trs_indexed_by_time[1])
        drawfn(ax, t, @lift(trs_indexed_by_time[$t + 1][i]))
    end
end

function draw_gt_and_particles(tr, particles)
    fig = Figure(resolution=(800, 1000))
    l1 = GridLayout()
    fig[1:2, 1] = l1
    l1[1, 1] = ax2d = Axis(fig[1, 1], aspect=DataAspect(), title="2D world")
    t = Observable(0)
    obs = draw_obs!(ax2d, t, tr)
    gt = draw_gt_sq!(ax2d, t, tr)
    draw_gt_occ!(ax2d, t, tr)
    inf = draw_particles_gt!(draw_particle_sq_gt!, ax2d, t, particles)
    xlims!(ax2d, (0.5, last(Positions()) + 0.5))

    l1[2, 1] = ax1d = Axis(fig[2, 1], aspect=DataAspect(), title="Inferred occluder position")
    hideydecorations!(ax1d)
    draw_particles_gt!(draw_particle_occ_gt!, ax1d, t, particles)
    linkxaxes!(ax2d, ax1d)

    rowsize!(l1, 1, Relative(19/20))
    rowsize!(l1, 2, Relative(1/20))
    # ax2d.height = Relative(20/21)
    # ax1d.height = Relative(1/21)
    # ax2d.tellheight = true
    # ax1d.tellheight = true
    # trim!(fig.layout)

    leg = Legend(fig[3, 1],
        [
            PolyElement(color=:gold),
            LineElement(color=:seagreen),
            PolyElement(color=:black)
        ],
        ["Observed image", "Ground-Truth Positions", "Inferred Positions"]
    )
    leg.orientation = :horizontal


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