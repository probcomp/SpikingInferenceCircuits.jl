using Gen
using ProbEstimates
include("model_hyperparameters.jl")
include("modeling_utils.jl")

### Obs model ###
# ╔═╡ d4c86af7-6292-4d34-b136-0e3e4e0660c2
p_got_photon(occ, sqx, sqy, x, y) = (
		is_occluded(occ, x, y) || is_in_square(sqx, sqy, x, y)
	) ? 1 - p_flip() : p_flip()

bern_probs(p) = [p, 1-p]
@gen (static) function render_pixel(a1, a2)
    (occ, sqx, sqy) = a1; (x, y) = a2
    got_photon ~ BoolCat(
        bern_probs(p_got_photon(occ, sqx, sqy, x, y))
    )
    return got_photon
end

@gen (static) function obs_model(occ, x, y, vx, vy)
    arg1, arg2 = Map2Dargs(
        fill((occ, x, y), (ImageSideLength(), ImageSideLength())),
        [(x, y) for x=1:ImageSideLength(), y=1:ImageSideLength()]
    )
    img_inner ~ Map2D(render_pixel)(arg1, arg2)
    return img_inner
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