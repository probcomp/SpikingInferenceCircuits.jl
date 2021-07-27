### UTIL ###
squarestart_given_center(val) = val - Int((SquareSideLength()-1)/2)

function prob_square_at_locations(img, occluder_pos)
    image_with_sq_pos(x, y) = image_determ(occluder_pos, x, y) .> 0
    error_with_sq_pos(x, y) = abs.(image_with_sq_pos(x, y) - img)

    n_flipped_with_sq_pos(x, y) = sum(error_with_sq_pos(x, y))
    log_likelihood_with_sq_pos(x, y) = 
        let nflip = n_flipped_with_sq_pos(x, y)
            nflip * log(p_flip()) + ImageSideLength()^2 * log(1 - p_flip())
        end

    log_likelihoods = [
        log_likelihood_with_sq_pos(x, y)
        for x in positions(SquareSideLength()),
            y in positions(SquareSideLength())
    ]
    probs = exp.(log_likelihoods .- logsumexp(log_likelihoods))
    return probs
end

function possible_square_locations(img)
    [ (squarestart_given_center(x), squarestart_given_center(y))
        for x=2:ImageSideLength()-2, y=2:ImageSideLength()-2
        if (
            sum(img[x-1:x+1, y-1:y+1]) ≥ 6 &&
            sum(img[x-1:x+1, :]) ≤ 30
        )
    ]
end

function x_probs(locations)
    counts = zeros(length(positions(SquareSideLength())))
    for (x, _) in locations
        counts[x] += 1
    end
    return counts/sum(counts)
end

function y_probs(selected_x, locations)
    counts = zeros(length(positions(SquareSideLength())))
    for (x, y) in locations
        if x == selected_x
            counts[y] += 1
        end
    end
    return counts/sum(counts)
end

occ_probs(img) = normalize([
    sum(img[x:x+OccluderLength(), :]) for x in positions(OccluderLength())	
])

### INITIAL PROPOSAL ###

vsd2(loc_probs) =vec(sum(loc_probs, dims=2))
@gen (static) function _initial_proposal(img)
    occₜ ~ Cat(occ_probs(img))
    loc_probs = prob_square_at_locations(img, occₜ)
    xₜ ~ Cat(vsd2(loc_probs))
    yₜ ~ Cat(normalize(loc_probs[xₜ, :]))
end

### STEP PROPOSAL ###
function x_step_probs(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, imgₜ, occₜ)
    x_probs_from_velocity = truncated_discretized_gaussian(
        xₜ₋₁ + vxₜ₋₁, 2., positions(SquareSideLength())
    )
    y_probs_from_velocity = truncated_discretized_gaussian(
        yₜ₋₁ + vyₜ₋₁, 2., positions(SquareSideLength())
    )

    probs_from_img = prob_square_at_locations(imgₜ, occₜ)
    x_probs_from_img = vec(sum(probs_from_img, dims=2))
    xprobs = normalize(x_probs_from_img.^2 .* x_probs_from_velocity)

    return (xprobs, (y_probs_from_velocity, probs_from_img))
end
function y_step_probs(x, (y_probs_from_velocity, probs_from_img))
    return normalize(probs_from_img[x, :].^2 .* y_probs_from_velocity)
end

function vel_probs(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, xₜ, yₜ)
    deltax = xₜ - xₜ₋₁
    deltay = yₜ - yₜ₋₁
    
    vx_probs = normalize((
        maybe_one_off(vxₜ₋₁, 0.4, Vels()) .*
        discretized_gaussian(deltax, 2., Vels())
    ))
    vy_probs = normalize((
        maybe_one_off(vyₜ₋₁, 0.4, Vels()) .*
        discretized_gaussian(deltay, 2., Vels())
    ))

    return (vx_probs, vy_probs)
end
function occ_probs(occₜ₋₁, new_img)
    oneoff_probs = maybe_one_off(
        occₜ₋₁, 0.6, positions(OccluderLength())
    )
    oprobs = normalize(occ_probs(new_img) .* oneoff_probs)
    oprobs = isapprox(sum(oprobs), 1.) ? oprobs : oneoff_probs
end

@gen (static) function _step_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, imgₜ)
    occₜ ~ Cat(normalize(truncate(occ_probs(occₜ₋₁, imgₜ))))

    xprobs, for_y = x_step_probs(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, imgₜ, occₜ)
    xₜ ~ Cat(normalize(truncate(xprobs)))
    yₜ ~ Cat(normalize(truncate(y_step_probs(xₜ, for_y))))

    vx_probs, vy_probs = vel_probs(xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, xₜ, yₜ)
    vxₜ ~ VelCat(normalize(truncate(vx_probs)))
    vyₜ ~ VelCat(normalize(truncate(vy_probs)))
end
# @gen (static) function step_proposal(prev_tr, new_img)
#     T = get_args(prev_tr)[1] + 1
#     prev_latents = get_submap(get_choices(prev_tr), latents_addr(T - 1))

#     {:steps => T => :latents} ~ _step_proposal(prev_latents, new_img)
# end
# latents_addr(t) = t == 0 ? :initial_latents : :steps => t => :latents