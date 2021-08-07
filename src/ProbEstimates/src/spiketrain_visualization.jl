module SpiketrainViz
using CairoMakie, Colors

export draw_spiketrain_figure, get_spiketrain_figure

"""
    draw_spiketrain_figure(
        lines; # Vector of either a String to display, or a Vector{Float64} of spiketimes for a spiketrain line
        labels=String[], # Vector of labels for the first `length(labels)` lines
        colors=Color[], # Vector of colors for the first `length(colors)` lines
        resolution=(1280, 720),
        figure_title="Spiketrain",
        time=0.,
        xmin=nothing, xmax=nothing, # min and max displayed x value
        xlabel="Time (ms)"
    )
"""
function draw_spiketrain_figure(args...; kwargs...)
    f = get_spiketrain_figure(args...; kwargs...)
    display(f)

    return f
end

function get_spiketrain_figure(
    lines; # Vector of either a String to display, or a Vector{Float64} of spiketimes for a spiketrain line
    labels=String[], # Vector of labels for the first `length(labels)` lines
    colors=Color[], # Vector of colors for the first `length(colors)` lines
    resolution=(1280, 720),
    figure_title="Spiketrain",
    time=0.,
    xmin=nothing, xmax=nothing, # min and max displayed x value
    xlabel="Time (ms)"
)
    f = Figure(;resolution)
    ax = f[1, 1] = Axis(f; title = figure_title, xlabel)

    draw_lines!(ax, lines, labels, colors, time, xmin, xmax)

    return f
end

function draw_lines!(ax, lines, labels, colors, time, xmin, xmax)
    lines, labels, colors = map(reverse, (lines, labels, colors))

    hideydecorations!(ax, ticklabels=false)
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
        draw_line!(ax, line, pos, trainheight, time, color)
    end

    xlims!(ax, compute_xlims(lines, xmin, xmax))
    ylims!(ax, (first(ypositions) - 1, last(ypositions) + 1))
    ax.yticks = (ypositions[1:length(labels)], labels)
    ax.yticklabelcolor[] = colors

    return nothing
end

infmin(vec) = (!(vec isa Vector) || isempty(vec)) ? Inf : minimum(vec)
infmax(vec) = (!(vec isa Vector) || isempty(vec)) ? -Inf : maximum(vec)
compute_xlims(trains, xmin, xmax) = (
    isnothing(xmin) ? minimum(map(infmin, trains)) : xmin,
    isnothing(xmax) ? maximum(map(infmax, trains)) : xmax
)

# Spiketrain line
function draw_line!(ax, spiketimes::Vector, ypos, height, current_time, color=RGB(0, 0, 0))
    @assert all(t isa Real for t in spiketimes) "a spiketimes vector (for a single y position) is not a vector of real numbers"
    y1 = ypos - height/2; y2 = ypos + height/2
    times = vcat([
        [Point2f0(t - current_time, y1), Point2f0(t - current_time, y2)]
        for t in spiketimes
    ]...)

    if !isempty(times)
        linesegments!(ax, times; color)
    end
end

# Text line
draw_line!(ax, text::String, ypos, height, current_time, color=RGB(0, 0, 0)) =
    text!(ax, text; position = (height, ypos), align = (:left, :center), color, textsize = 16*height)

end