includet("../model/model.jl")
includet("../groundtruth_rendering.jl")
includet("../visualize.jl")
includet("../run_utils.jl")

ProbEstimates.use_perfect_weights!()

using DynamicModels: nest_at, obs_choicemap, latents_choicemap

@load_generated_functions

function sample_without_replacement(n, set)
    c = collect(set)
    samples = []
    for _=1:n
        i = uniform_discrete(1, length(c))
        while (c[i] in samples)
            i = uniform_discrete(1, length(c))
        end
        push!(samples, c[i])
    end
    return samples
end
function get_tr_with_n_flips(gt_tr, n)
    flip_positions = sample_without_replacement(n, [(x, y) for x in 1:ImageSideLength(), y in 1:ImageSideLength()])
    flips_choicemap = choicemap(
        Iterators.flatten(
            (
                (:init => :obs => :img_inner => x => y => :pixel_color => :flip1 => :val, ((x, y) in flip_positions)),
                (:init => :obs => :img_inner => x => y => :pixel_color => :flip2 => :val, ((x, y) in flip_positions))
            )
            for x in 1:ImageSideLength() for y in 1:ImageSideLength()
        )...
    )
    tr, _ = generate(model, (0,), merge(get_selected(get_choices(gt_tr), select(:init => :latents)), flips_choicemap))
    # @assert latents_choicemap(tr, 0) == latents_choicemap(gt_tr, 0)
    return tr
end

function get_trs_with_various_flipcounts(flipcounts, gt_tr=simulate(model, (0,)))
    return [get_tr_with_n_flips(gt_tr, cnt) for cnt in flipcounts]
end

function draw_representative_imgs_square!(f, figpos, counts=[0, 2, 4, 6])
    @assert length(counts) == 4

    Label(f.layout[figpos..., Top()],
            "Images with Different Numbers\nof Pixel Flips",
            padding = (0, 0, 25, 0)
        )

    layout = f[figpos...] = GridLayout()
    trs = get_trs_with_various_flipcounts(counts)

    for (i, (x, y)) in enumerate([(x, y) for y=1:2, x=1:2])
        ax = Axis(layout[x, y], title="$(counts[i]) Flips", aspect=DataAspect())
        hidedecorations!(ax)
        draw_obs!(ax, Observable(0), (trs[i]))
    end
end

function draw_representative_imgs_line!(f, figpos, counts=[0, 2, 4, 6])
    layout = f[figpos...] = GridLayout()
    trs = get_trs_with_various_flipcounts(counts)

    for i=1:length(counts)
        ax = Axis(layout[1, i], aspect=DataAspect())
        Label(layout[1, i, Bottom()], "$(counts[i]) flips", padding=(0,0,0,-20))
        hidedecorations!(ax)
        draw_obs!(ax, Observable(0), (trs[i]))
    end

    Label(layout[1, :, Bottom()],
        "Images with Different Numbers of Pixel Flips",
        padding=(0,0,0,40),
        valign=:top
    )
end