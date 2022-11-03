module SpiketrainViz
using CairoMakie, Colors
import ..Spiketrains

export draw_spiketrain_figure, get_spiketrain_figure, draw_spiketrain_figure_animated

get_color(::Spiketrains.VarValLine) = VAR_VAL_COLOR()
get_color(spec::Spiketrains.ScoreLine) = spec.do_recip_score ? RECIP_SCORE_COLOR() : FWD_SCORE_COLOR()
get_color(::Spiketrains.NormalizedWeight) = PARTICLE_WEIGHT_COLOR()
get_color(::Spiketrains.LogNormalization) = AUTONORM_COLOR()
get_color(s::Spiketrains.SubsidiarySingleParticleLineSpec) = get_color(s.spec)
get_color(s::Spiketrains.DistLine) = s.is_p ? P_DIST_COLOR() : Q_DIST_COLOR()
get_colors(groups::Vector{<:Union{Spiketrains.LabeledSingleParticleLineGroup, Spiketrains.LabeledMultiParticleLineGroup}}) =
    get_colors(reduce(vcat, g.line_specs for g in groups))
get_colors(lines) = map(get_color, lines)

rgbhex(r, g, b) = RGB(r/256, g/256, b/256)

# RECIP_SCORE_COLOR() = colorant"navy" # rgbhex(107, 47, 85) # rgbhex(229, 181, 211) # 
# FWD_SCORE_COLOR() = colorant"red" # rgbhex(48, 133, 133) # RGB(194/256, 230/256, 230/256) # colorant"red"
# VAR_VAL_COLOR() = colorant"green"
# PARTICLE_WEIGHT_COLOR() = colorant"coral"
# AUTONORM_COLOR() = colorant"indigo" #  colorant"violet"
RECIP_SCORE_COLOR() = colorant"black"
FWD_SCORE_COLOR() = colorant"black"
VAR_VAL_COLOR() = colorant"black"
PARTICLE_WEIGHT_COLOR() = colorant"black"
AUTONORM_COLOR() = colorant"black"
P_DIST_COLOR() = colorant"black"
Q_DIST_COLOR() = colorant"black"

"""
    draw_spiketrain_figure(
        lines; # Vector of either a String to display, or a Vector{Float64} of spiketimes for a spiketrain line
        labels=String[], # Vector of labels for the first `length(labels)` lines
        colors=Color[], # Vector of colors for the first `length(colors)` lines
        group_labels, # list of (label::String, num_lines::Int) for each group [each spiketrain is assumed to be in a group]
        resolution=(1280, 720),
        figure_title="Spiketrain",
        time=0.,
        xmin=0., xmax=nothing, # min and max displayed x value
        xlabel="Time (ms)"
    )
"""
function draw_spiketrain_figure(args...; kwargs...)
    f = get_spiketrain_figure(args...; kwargs...)
    display(f)

    return f
end
function draw_spiketrain_figure_animated(args...; kwargs...)
    t = Observable(0.)
    f = get_spiketrain_figure(args...; kwargs..., time=t)
    display(f)

    return (f, t)
end

#=
group_labels[i] describes the labels for the `i`th level of grouping.
group_labels[i] = (labels_and_lengths, offset_from_axis)
`offset_from_axis` is the number of points to the right of the axis to draw this label.
`labels_and_lengths` is an iterator over pairs `(label, length)` giving the label for a group
and the number of primitive lines in the group.
=#
function get_spiketrain_figure(
    lines; # Vector of either a String to display, or a Vector{Float64} of spiketimes for a spiketrain line
    labels=String[], # Vector of labels for the first `length(labels)` lines
    colors=Color[], # Vector of colors for the first `length(colors)` lines
    group_labels,
    resolution=(1280, 720),
    figure_title="Spiketrain",
    time=0.,
    xmin=0., xmax=nothing, # min and max displayed x value
    xlabel="Time (ms)"
)
    f = Figure(;resolution)
    ax = f[1, 1] = Axis(f; title = figure_title, xlabel)

    draw_lines!(ax, lines, labels, colors, time, xmin, xmax)
    for (labels_and_lengths, offset_from_axis) in group_labels
        draw_group_labels!(f, ax, labels_and_lengths, offset_from_axis, colors)
    end
    ax.yticklabelsvisible=false

    return f
end

draw_group_labels!(f, ax, group_labels, offset_from_axis, colors) = draw_group_labels!(f, f.layout, ax, group_labels, offset_from_axis, colors)
function draw_group_labels!(f, layout, ax, group_labels, offset_from_axis, colors)
    colsize!(layout, 1, Relative(0.6))
    println("group_labels = $group_labels")
    endpoint_indices = get_group_endpoint_indices(group_labels)

    bot = 1
    group_colors = []
    for (_, len) in group_labels
        colors_in_range = Set(colors[bot:(bot + len - 1)])
        if length(colors_in_range) != 1
            println("colors_in_range = $colors_in_range")
            push!(group_colors, colorant"black")
        else
            push!(group_colors, only(colors_in_range))
        end
        bot += len
    end

    ax.yticks = 1:first(endpoint_indices)[1]
    
    rhs(pos, px_area) = Point2((px_area.origin + px_area.widths)[1], pos[2])
    brackets = [
        lift(ax.elements[:yaxis].tickpositions, ax.scene.px_area) do pos, p
            ydiff = pos[2][2] - pos[1][2]
            y_increase = (ydiff / 2) * 0.8
            [
                rhs(pos[st], p) + Point2(offset_from_axis, y_increase), rhs(pos[st], p) + Point2(offset_from_axis + 5, y_increase),
                rhs(pos[st], p) + Point2(offset_from_axis + 5, y_increase), rhs(pos[nd], p) + Point2(offset_from_axis + 5, -y_increase),
                rhs(pos[nd], p) + Point2(offset_from_axis + 5, -y_increase), rhs(pos[nd], p) + Point2(offset_from_axis, -y_increase)
            ]
        end
        for (st, nd) in endpoint_indices
    ]
    for (color, bracketpoints) in zip(group_colors, brackets)
        linesegments!(f.scene, bracketpoints; color)
    end

    textpositions = [
        lift(ax.elements[:yaxis].tickpositions, ax.scene.px_area) do pos, p
            (
                (p.origin + p.widths)[1] + 10 + offset_from_axis,
                (pos[st][2] + pos[nd][2]) / 2
            )
        end
        for (st, nd) in endpoint_indices
    ]
    for ((label, _), pos, color) in zip(group_labels, textpositions, group_colors)
        text!(f.scene, label; position=pos, align=(:left, :center), textsize=15, color)
    end
end

"""
Returns a list of `(startidx, endidx)` for each group.  Corresponds to index locations
on the image (which are in the opposite order of the given lines.)
"""
function get_group_endpoint_indices(group_labels)    
    idx = 0
    idxpairs = []
    for (_, num_lines) in group_labels
        push!(idxpairs, (idx + 1, idx + num_lines))
        idx += num_lines
    end

    # reverse the indexing before returning
    return [(idx - st + 1, idx - nd + 1) for (st, nd) in idxpairs]
end

function draw_lines!(ax, lines, labels, colors, time, xmin, xmax; hide_y_decorations=true)
    if !(time isa Observable)
        time = Observable(time)
    end

    lines, labels, colors = map(reverse, (lines, labels, colors))

    if hide_y_decorations
        hideydecorations!(ax, ticklabels=false)
    end
    if isempty(lines) 
        @warn "Lines was empty; not drawing anything."
        return nothing;
    end

    # set neuron names on axis label
    ypositions = 1:length(lines)
    trainheight = 1

    colors = collect(Iterators.flatten((colors, Iterators.repeated(RGBA(0, 0, 0, 1), length(lines) - length(colors)))))
    @assert length(lines) == length(colors)

    for (line, pos, color) in zip(lines, ypositions, colors)
        draw_line!(ax, line, pos, trainheight, time, color; n_lines=length(lines))
    end

    xlims!(ax, (time[], time[] + 50))
    onany(time) do t # update the limits at the given times
        xlims!(ax, (t[], t[] + 50))
    end
    ylims!(ax, (first(ypositions) - 1, last(ypositions) + 1))

    if !isempty(labels)
        ax.yticks = (ypositions[1:length(labels)], labels)
        ax.yticklabelcolor[] = colors
    end

    return nothing
end

infmin(vec) = (!(vec isa Vector) || isempty(vec)) ? Inf : minimum(vec)
infmax(vec) = (!(vec isa Vector) || isempty(vec)) ? -Inf : maximum(vec)
function compute_xlims(trains, xmin, xmax)
    bot = isnothing(xmin) ? minimum(map(infmin, trains)) : xmin
    top = if isnothing(xmax)
            lastval = maximum(map(infmax, trains))
            bot + (lastval - bot)*1.05 # have a little extra space
        else
            xmax
        end
    return (bot, top)
end

# Spiketrain line
function draw_line!(ax, spiketimes::Vector, ypos, height, current_time, color=RGB(0, 0, 0);
    drawpoints=false, n_lines=nothing,
    minheight= isnothing(n_lines) ? 2 : n_lines/200
)
    @assert all(t isa Real for t in spiketimes) "a spiketimes vector (for a single y position) is not a vector of real numbers"
    
    if !(current_time isa Observable)
        current_time = Observable(current_time)
    end

    if drawpoints
        # times = @lift(Point2[Point2(t - $current_time, ypos) for t in spiketimes])
        times = Point2[Point2(t, ypos) for t in spiketimes]
        scatter!(ax, times; color, markersize=2)
    else
        height = max(minheight, height)*2
        y1 = ypos - height/2
        y2 = ypos + height/2
        # times = @lift(vcat([
        #     [Point2(t - $current_time, y1), Point2(t - $current_time, y2)]
        #     for t in spiketimes
        # ]...))
        times = vcat([
            [Point2(t, y1), Point2(t, y2)]
            for t in spiketimes
        ]...)

        if !isempty(times)
            linesegments!(ax, times; color, linewidth=2)
        end
    end
end

# Text line
draw_line!(ax, text::String, ypos, height, current_time, color=RGB(0, 0, 0); n_lines=nothing) =
    text!(ax, text; position = (height, ypos), align = (:left, :center), color, textsize = 16*height)

end