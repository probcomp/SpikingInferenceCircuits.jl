using Gen
using ProbEstimates
include("model_hyperparameters.jl")
include("modeling_utils.jl")

abstract type PixelColor end
struct Empty <: PixelColor; end
struct Object <: PixelColor; end
struct Occluder <: PixelColor; end
PixelColors() = [Empty(), Object(), Occluder()]

bern_probs(p) = [p, 1-p]

@gen (static) function is_in_square(squarex, squarey, x, y)
    x_in_range = squarex ≤ x ≤ squarex + SquareSideLength() - 1
    y_in_range = squarey ≤ y ≤ squarey + SquareSideLength() - 1
    return x_in_range && y_in_range
end

@gen (static) function render_pixel(occ, sqx, sqy, x, y)
    is_occluded = occ ≤ x ≤ occ + OccluderLength() - 1
    in_sq ~ is_in_square(sqx, sqy, x, y)
    color = is_occluded ? Occluder() : in_sq ? Object() : Empty()
    pixel_color ~ LCat(PixelColors())(onehot(color, PixelColors()))
    return pixel_color
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

vel_change_probs(vxₜ₋₁, xₜ₋₁) =
    if (xₜ₋₁ ≤ first(SqPos()) && vxₜ₋₁ < 0) || (xₜ₋₁ ≥ last(SqPos()) && vxₜ₋₁ > 0)
        maybe_one_off(-vxₜ₋₁, VelOneOffProb(), Vels())
    else
        maybe_one_off(vxₜ₋₁, VelOneOffProb(), Vels())
    end
@gen (static) function step_latent_model(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)
    vxₜ ~ VelCat(vel_change_probs(vxₜ₋₁, xₜ₋₁))
    vyₜ ~ VelCat(vel_change_probs(vyₜ₋₁, yₜ₋₁))
	occₜ ~ Cat(maybe_one_off(occₜ₋₁, OccOneOffProb(), positions(OccluderLength())))
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, positions(SquareSideLength())))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, positions(SquareSideLength())))
	# xₜ ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + vxₜ, 2., positions(SquareSideLength())))
	# yₜ ~ Cat(truncated_discretized_gaussian(yₜ₋₁ + vyₜ, 2., positions(SquareSideLength())))
	return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end