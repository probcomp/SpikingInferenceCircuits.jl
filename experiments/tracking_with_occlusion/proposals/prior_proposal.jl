@gen (static) function _initial_prior_proposal(img_inner)
    occₜ ~ Cat(uniform(positions(OccluderLength())))
    xₜ ~ Cat(uniform(positions(SquareSideLength())))
    yₜ ~ Cat(uniform(positions(SquareSideLength())))
    vxₜ ~ VelCat(uniform(Vels()))
    vyₜ ~ VelCat(uniform(Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
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
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

initial_prior_proposal = @compile_initial_proposal(_initial_prior_proposal, obs_aux_proposal, 5, 1)
step_prior_proposal = @compile_step_proposal(_step_prior_proposal, obs_aux_proposal, 5, 1)