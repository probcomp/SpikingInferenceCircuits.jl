#=
img[x][y] gives a PixelColor.
=#

function get_x_pos(img)
    xs = findall([any(pix == Object() for pix in col) for col in img])
    if length(xs) == 0
        return nothing
    else
        return only(xs)
    end
end
get_y_pos(img, ::Nothing) = nothing
get_y_pos(img, x) = get_y_pos(img[x])
get_y_pos(col) = only(findall([pix == Object() for pix in col]))
get_occluder_pos(img) = first(findall([any(pix == Occluder() for pix in col) for col in img]))

occluder_informed_x_prior(occ) = [occ ≤ x ≤ occ + OccluderLength() - 1 ? 1. : 0. for x in SqPos()] |> normalize
uninformed_y_prior() = unif(SqPos())
@gen (static) function _init_proposal(img)
    occluder_pos = get_occluder_pos(img)
    occₜ ~ Cat(onehot(occluder_pos, OccPos()))

    x_pos = get_x_pos(img)
    y_pos = get_y_pos(img, x_pos)    
    xₜ ~ Cat(
        isnothing(x_pos) ? occluder_informed_x_prior(occₜ) : onehot(x_pos, SqPos())
    )
    yₜ ~ Cat(
        isnothing(y_pos) ? uninformed_y_prior() : onehot(y_pos, SqPos())
    )

    vxₜ ~ VelCat(uniform(Vels()))
	vyₜ ~ VelCat(uniform(Vels()))
end

vel_to_idx(vel) = vel - first(Vels()) + 1
vel_probs_to_x_probs(v_probs, xₜ₋₁) =
    sum(
        v_probs[vel_to_idx(v)] * onehot(xₜ₋₁ + v, SqPos())
        for v in Vels()
    )
function step_x_dist(occₜ, vxₜ₋₁, xₜ₋₁) # TODO: there's a bug!!
    # println((occₜ, xₜ₋₁, vxₜ₋₁))
    v_probs = vel_change_probs(vxₜ₋₁, xₜ₋₁)
    @assert sum(v_probs) > 0
    x_probs_from_vel = vel_probs_to_x_probs(v_probs, xₜ₋₁)
    occ_probs = occluder_informed_x_prior(occₜ)
    unnormalized_probs = x_probs_from_vel .* occ_probs
    if sum(unnormalized_probs) > 0
        return normalize(unnormalized_probs)
    else # then this trace will end up having 0 probability
        return onehot(1, SqPos()) # v_probs
    end
end
step_y_dist(vyₜ₋₁, yₜ₋₁) = vel_change_probs(vyₜ₋₁, yₜ₋₁) |> vp -> vel_probs_to_x_probs(vp, yₜ₋₁) |> normalize
function vel_step_dist(xₜ, xₜ₋₁, vxₜ₋₁)
    v_probs = vel_change_probs(vxₜ₋₁, xₜ₋₁)
    x_likelihoods = [onehot(xₜ₋₁ + v, Positions())[xₜ] for v in Vels()]
    unnormalized_probs = v_probs .* x_likelihoods
    if sum(unnormalized_probs) > 0
        return normalize(unnormalized_probs)
    else # trace will have 0 probability; can return any pvec
        return onehot(last(Vels()), Vels())
    end
end
@gen (static) function _step_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    occluder_pos = get_occluder_pos(img)
    occₜ ~ Cat(onehot(occluder_pos, OccPos()))

    x_pos = get_x_pos(img)
    y_pos = get_y_pos(img, x_pos)    
    xₜ ~ Cat(
        isnothing(x_pos) ? step_x_dist(occₜ, vxₜ₋₁, xₜ₋₁) : onehot(x_pos, SqPos())
    )
    yₜ ~ Cat(
        isnothing(y_pos) ? step_y_dist(vyₜ₋₁, yₜ₋₁) : onehot(y_pos, SqPos())
    )

    vxₜ ~ VelCat(vel_step_dist(xₜ, xₜ₋₁, vxₜ₋₁))
    vyₜ ~ VelCat(vel_step_dist(yₜ, yₜ₋₁, vyₜ₋₁))
end