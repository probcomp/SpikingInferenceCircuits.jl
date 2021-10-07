function occ_likelihood(img, x)
    num_correct = sum([pix == Occluder() ? 1 : 0 for pix in Iterators.flatten(img[x:(x + OccluderLength() - 1)])])
    num_incorrect = OccluderLength() * size(img)[1] - num_correct
    return (1 - ColorFlipProb())^num_correct * (ColorFlipProb() * 1/2)^(num_incorrect)
end
function object_likelihood(img, (x, y))
    is_correct = img[x][y] == Object()
    return is_correct ? (1 - ColorFlipProb()) : 1/2 * ColorFlipProb()
end
init_occ_probs(img) = normalize([occ_likelihood(img, x) for x in positions(OccluderLength())])
init_ball_probs(img) = normalize([
    object_likelihood(img, (x, y))
    for x in positions(SquareSideLength()),
        y in positions(SquareSideLength())
])

sum_over_y(matrix) = reshape(sum(matrix, dims=2), (:,))
y_for_x(matrix, x) = reshape(matrix[x, :], (:,))
@gen (static) function _init_near_locopt_proposal(img)
    occₜ ~ Cat(init_occ_probs(img))

    xyprobs = init_ball_probs(img)

    xₜ ~ Cat(normalize(sum_over_y(xyprobs)))
    yₜ ~ Cat(normalize(y_for_x(xyprobs, xₜ)))

    vxₜ ~ VelCat(uniform(Vels()))
	vyₜ ~ VelCat(uniform(Vels()))

    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

xy_to_vel_likelihoods(xylikelihoods, xₜ₋₁, yₜ₋₁) = [
    xylikelihoods[
        truncate_value(xₜ₋₁ + vx, positions(SquareSideLength())),
        truncate_value(yₜ₋₁ + vy, positions(SquareSideLength()))
    ]
    for vx in Vels(), vy in Vels()
]

pointwise_product(x, y) = x .* y
@gen (static) function _step_near_locopt_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    occₜ ~ Cat(
        normalize(
            pointwise_product(
                init_occ_probs(img),
                discretized_gaussian(occₜ₋₁, OccOneOffProb(), positions(OccluderLength()))
            )
        )
    )

    xylikelihoods = init_ball_probs(img)
    vel_likelihoods = xy_to_vel_likelihoods(xylikelihoods, xₜ₋₁, yₜ₋₁)

    vxₜ ~ VelCat(normalize(
        pointwise_product(sum_over_y(vel_likelihoods), vel_change_probs(vxₜ₋₁, xₜ₋₁))
    ))
    vyₜ ~ VelCat(normalize(
        pointwise_product(y_for_x(vel_likelihoods, vxₜ - first(Vels()) + 1), vel_change_probs(vyₜ₋₁, yₜ₋₁))
    ))

    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, positions(SquareSideLength())))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, positions(SquareSideLength())))

    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

initial_near_locopt_proposal = @compile_initial_proposal(_init_near_locopt_proposal, obs_aux_proposal, 5, 1)
step_near_locopt_proposal = @compile_step_proposal(_step_near_locopt_proposal, obs_aux_proposal, 5, 1)