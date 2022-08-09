using Gen
using ProbEstimates
using DynamicModels
include("model_hyperparameters.jl")
include("modeling_utils.jl")


abstract type PixelColor end
struct Empty <: PixelColor; end
struct Object <: PixelColor; end
struct Occluder <: PixelColor; end
PixelColors() = [Empty(), Object(), Occluder()]
ColorFlipProb() = 0.01

global use_aux_vars = true
function set_use_aux_vars!(val)
    global use_aux_vars = val
end
flip1_prob() = use_aux_vars ? sqrt(ColorFlipProb()) : ColorFlipProb()
flip2_prob(flip1) = use_aux_vars ? sqrt(ColorFlipProb()) : (flip1 ? 1.0 : 0.0)

bern_probs(p) = [p, 1-p]

@gen (static) function is_in_square(squarex, squarey, x, y)
    x_in_range = squarex ≤ x ≤ squarex + SquareSideLength() - 1
    y_in_range = squarey ≤ y ≤ squarey + SquareSideLength() - 1
    return x_in_range && y_in_range
end

uniform_from_other_colors(color) = normalize([c == color ? 0. : 1. for c in PixelColors()])
colorprobvec(expected_color) = [c == expected_color ? 1 - ColorFlipProb() : 1/2 * ColorFlipProb() for c in PixelColors()]
@gen function maybe_flip_color(expected_color, x, y)
    flip1 ~ BoolCat(bern_probs(flip1_prob()))
    flip2 ~ BoolCat(bern_probs(flip2_prob(flip1)))
    color ~ LCat(PixelColors())(
        # colorprobvec(expected_color)
        flip1 && flip2 ? uniform_from_other_colors(expected_color) : onehot(expected_color, PixelColors())
    )

    if color != expected_color
        @assert flip1 && flip2
    end
    if color == expected_color
        @assert !flip1 || !flip2 "color = $color ; expected_color = $expected_color; flip1 = $flip1 ; flip2 = $flip2; (x, y) = $((x, y))"
    end

    return color
end

@gen (static) function render_pixel(occ, sqx, sqy, x, y)
    is_occluded = occ ≤ x ≤ occ + OccluderLength() - 1
    in_sq ~ is_in_square(sqx, sqy, x, y)
    color = is_occluded ? Occluder() : in_sq ? Object() : Empty()
    pixel_color ~ maybe_flip_color(color, x, y)
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
        xs(),
        fill(1:ImageSideLength(), ImageSideLength())
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
        discretized_gaussian(-vxₜ₋₁, VelStd(), Vels())
    else
        discretized_gaussian(vxₜ₋₁, VelStd(), Vels())
    end
@gen (static) function step_latent_model(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁)
    vxₜ ~ VelCat(vel_change_probs(vxₜ₋₁, xₜ₋₁))
    vyₜ ~ VelCat(vel_change_probs(vyₜ₋₁, yₜ₋₁))
	occₜ ~ Cat(discretized_gaussian(occₜ₋₁, OccOneOffProb(), positions(OccluderLength())))
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, positions(SquareSideLength())))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, positions(SquareSideLength())))
	# xₜ ~ Cat(truncated_discretized_gaussian(xₜ₋₁ + vxₜ, 2., positions(SquareSideLength())))
	# yₜ ~ Cat(truncated_discretized_gaussian(yₜ₋₁ + vyₜ, 2., positions(SquareSideLength())))
	return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
