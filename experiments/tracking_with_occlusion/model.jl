using Gen
using ProbEstimates
include("model_hyperparameters.jl")
include("modeling_utils.jl")

### Obs model ###
# ╔═╡ d4c86af7-6292-4d34-b136-0e3e4e0660c2
# p_got_photon(occ, sqx, sqy, x, y) = (
# 		is_occluded(occ, x, y) || is_in_square(sqx, sqy, x, y)
# 	) ? 1 - p_flip() : p_flip()

# @gen (static) function is_colored(occ, sqx, sqy, x, y)
# 	isocc = is_occluded(occ, x)
# 	x_in_sq = sqx ≤ x ≤ sqx + SquareSideLength()
# 	y_in_sq = sqy ≤ y ≤ sqy + SquareSideLength()
# 	in_sq = x_in_sq && y_in_sq
# 	square_colored = isocc || in_sq
# 	return square_colored
# end

# bern_probs(p) = [p, 1-p]
# @gen (static) function render_pixel(occ, sqx, sqy, x, y)
# 	square_colored ~ is_colored(occ, sqx, sqy, x, y)
#     got_photon ~ BoolCat(
#         bern_probs(square_colored ? 1 - p_flip() : p_flip())
#     )
#     return got_photon
# end

# fill_2dmap(v) = Map2Dargs(fill(v, (ImageSideLength(), ImageSideLength())))
# @gen (static) function obs_model(occ, x, y, vx, vy)
# 	occgrid = fill_2dmap(occ)
# 	sqxgrid = fill_2dmap(x)
# 	sqygrid = fill_2dmap(y)
# 	xgrid   = Map2Dargs([x for x=1:ImageSideLength(), y=1:ImageSideLength()])
# 	ygrid   = Map2Dargs([y for x=1:ImageSideLength(), y=1:ImageSideLength()])

#     img_inner ~ Map2D(render_pixel)(occgrid, sqxgrid, sqygrid, xgrid, ygrid)
#     return img_inner
# end

@gen (static) function is_in_square(squarex, squarey, x, y)
    x_in_range = squarex ≤ x ≤ squarex + SquareSideLength()
    y_in_range = squarey ≤ y ≤ squarey + SquareSideLength()
    return x_in_range && y_in_range
end
@gen (static) function pixel_expected_on(occ, sqx, sqy, x, y)
    is_occluded = occ ≤ x ≤ OccluderLength()
    in_sq ~ is_in_square(sqx, sqy, x, y)
    return is_occluded || in_sq
end

@gen (static) function render_pixel(occ, sqx, sqy, x, y)
    expect_pixel_on ~ pixel_expected_on(occ, sqx, sqy, x, y)
    got_photon ~ bernoulli(expect_pixel_on ? 1 - p_flip() : p_flip())
    return got_photon
end

xs() = [[pixx for _=1:ImageSideLength()] for pixx=1:ImageSideLength()]
@gen (static) function obs_model(occ, x, y, vx, vy)
    # TODO: Figure out whether each pixel is in the given range
    # _before_ calling `fill` to reduce the number of edges
    img_inner ~ Map(Map(render_pixel))(
        fill(fill(occ, ImageSideLength()), ImageSideLength()),
        fill(fill(x, ImageSideLength()), ImageSideLength()),
        fill(fill(y, ImageSideLength()), ImageSideLength()),
        fill(1:ImageSideLength(), ImageSideLength()),
        xs()
    )

    return (img_inner,)
end

### initial & step model ###
### initial & step model ###
@gen (static) function init_latent_model()
	occₜ ~ Cat(uniform(positions(OccluderLength())))
	xₜ ~ Cat(uniform(positions(SquareSideLength())))
	yₜ ~ Cat(uniform(positions(SquareSideLength())))
	vxₜ ~ VelCat(uniform(Vels()))
	vyₜ ~ VelCat(uniform(Vels()))
	return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end
@gen (static) function step_latent_model(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)
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