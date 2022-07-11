module SpiketrainViz
using GLMakie, Colors
GLMakie.inline!(false)

export draw_spiketrain_figure, get_spiketrain_figure

"""
    draw_spiketrain_figure(
        spiketrains; # Vector of `Vector{Float64}`s
        names=String[], # Vector of neuron names for the first `length(names)` spiketrains
        colors=Color[], # Vector of colors for the first `length(colors)` spiketrains
        resolution=(1280, 720),
        figure_title="Spiketrain",
        time=0.,
        xmin=nothing,
        xmax=nothing
    )
"""
function draw_spiketrain_figure(args...; kwargs...)
    f = get_spiketrain_figure(args...; kwargs...)
    display(f)

    return f
end

function get_spiketrain_figure(
    spiketrains; # Vector of `Vector{Float64}`s
    names=String[], # Vector of neuron names for the first `length(names)` spiketrains
    colors=Color[], # Vector of colors for the first `length(colors)` spiketrains
    resolution=(1280, 720),
    figure_title="Spiketrain",
    time=0.,
    xmin=nothing, xmax=nothing # min and max displayed x value
)
    f = Figure(;resolution)
    ax = f[1, 1] = Axis(f; title = figure_title)

    draw_spiketrain!(ax, spiketrains, names, colors, time, xmin, xmax)

    return f
end

function draw_spiketrain!(ax, spiketrains, names, colors, time, xmin, xmax)
    hideydecorations!(ax, ticklabels=false)
    if isempty(spiketrains) 
        return nothing;
    end

    # set neuron names on axis label
    ypositions = 1:length(spiketrains)
    trainheight = 1

    colors = collect(Iterators.flatten((colors, Iterators.repeated(RGBA(0, 0, 0, 1), length(names) - length(colors)))))
    @assert length(names) == length(colors)

    for (spiketrain, pos, color) in zip(spiketrains, ypositions, colors)        
        draw_single_spiketrain!(ax, spiketrain, pos, trainheight, time, color)
    end

    xlims!(ax, compute_xlims(spiketrains, xmin, xmax))
    ylims!(ax, (first(ypositions) - 1, last(ypositions) + 1))
    ax.yticks = (ypositions[1:length(names)], names)
    ax.yticklabelcolor[] = colors

    return nothing
end

infmin(vec) = isempty(vec) ? Inf : minimum(vec)
infmax(vec) = isempty(vec) ? -Inf : maximum(vec)
compute_xlims(trains, xmin, xmax) = (
    isnothing(xmin) ? minimum(map(infmin, trains)) : xmin,
    isnothing(xmax) ? maximum(map(infmax, trains)) : xmax
)

function draw_single_spiketrain!(ax, spiketimes, ypos, height, current_time, color=RGBA(0, 0, 0, 1))
    @assert all(t isa Real for t in spiketimes) "a spiketimes vector (for a single y position) is not a vector of real numbers"
    y1 = ypos - height/2; y2 = ypos + height/2
    times = vcat([
        [Point2(t - current_time, y1), Point2(t - current_time, y2)]
        for t in spiketimes
    ]...)

    if !isempty(times)
        linesegments!(ax, times; color)
    end
end

end