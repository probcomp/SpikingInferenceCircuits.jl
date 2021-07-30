@gen (static) function _initial_prior_proposal(img_inner)
    occₜ ~ Cat(uniform(positions(OccluderLength())))
    xₜ ~ Cat(uniform(positions(SquareSideLength())))
    yₜ ~ Cat(uniform(positions(SquareSideLength())))
    vxₜ ~ VelCat(uniform(Vels()))
    vyₜ ~ VelCat(uniform(Vels()))
end
@gen (static) function _step_prior_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img_inner)
    occₜ ~ Cat(maybe_one_off(occₜ₋₁, 0.3, positions(OccluderLength())))
    xₜ ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + vxₜ₋₁, 2.,
            positions(SquareSideLength()))
    )
    yₜ ~ Cat(truncated_discretized_gaussian(yₜ₋₁ + vyₜ₋₁, 2.,
        positions(SquareSideLength()))
    )
    vxₜ ~ VelCat(maybe_one_off(vxₜ₋₁, 0.4, Vels()))
    vyₜ ~ VelCat(maybe_one_off(vyₜ₋₁, 0.4, Vels()))
end