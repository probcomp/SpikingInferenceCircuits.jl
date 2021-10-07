# fill in the auxiliary variables in the observation model
prob_flip1_given_no_flip() =
    if use_aux_vars
        let p = ColorFlipProb()
            (√(p) - p) / (1 - p)
            # √(p)
        end
    else
        0.
    end
prob_flip2_given_no_flip(flip1) =
    if use_aux_vars
        flip1 ? 0.0 : √(ColorFlipProb())
    else
        0.
    end

@gen (static) function _fill_in_flips(is_correct)
    flip1 ~ BoolCat(bern_probs(
        is_correct ? prob_flip1_given_no_flip() : 1.0
    ))
    flip2 ~ BoolCat(bern_probs(
        is_correct ? prob_flip2_given_no_flip(flip1) : 1.0)
    )
end
# @gen (static) function _fill_in_flips(is_correct)
#     flip1 ~ BoolCat(bern_probs(
#         is_correct ? flip1_prob() : 1.0
#     ))
#     flip2 ~ BoolCat(bern_probs(
#         is_correct ? flip2_prob(flip1) : 1.0)
#     )
# end
@gen (static) function fill_in_flips(is_correct)
    {:pixel_color} ~ _fill_in_flips(is_correct)
end
function is_flipped_vec_of_vecs(determ_img, img)
    # display(determ_img)
    # display(img)
    grid = [
        [
            determ_img[x, y] == img[x][y]
            for y in positions(SquareSideLength())
        ]
        for x in positions(SquareSideLength())
    ]
    # sizes = [length(map(x -> x ? 1 : 0, line)) for line in grid]
    # @assert sum(sizes) > (length(determ_img) * 0.8)
    # display(grid)
    return grid
end
@gen (static) function obs_aux_proposal(occₜ, xₜ, yₜ, vxₜ, vyₜ, img)
    determ_img = img_determ_with_colors(occₜ, xₜ, yₜ)
    img_inner ~ Map(Map(fill_in_flips))(is_flipped_vec_of_vecs(determ_img, img))
end