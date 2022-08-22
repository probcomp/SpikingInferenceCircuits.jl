includet("model.jl")

expanded_color_dist(expected_color) = ColorFlipProb() * uniform_from_other_colors(expected_color) + (1 - ColorFlipProb()) * onehot(expected_color, PixelColors())
@gen (static) function maybe_flip_color_noflipvars(expected_color)
    color ~ LCat(PixelColors())(
        expanded_color_dist(expected_color)
    )
    return color
end

@gen (static) function render_pixel_noflipvars(occ, sqx, sqy, x, y)
    is_occluded = occ ≤ x ≤ occ + OccluderLength() - 1
    in_sq ~ is_in_square(sqx, sqy, x, y)
    color = is_occluded ? Occluder() : in_sq ? Object() : Empty()
    pixel_color ~ maybe_flip_color_noflipvars(color)
    return pixel_color
end

xs() = [[pixx for _=1:ImageSideLength()] for pixx=1:ImageSideLength()]
@gen (static) function obs_model_noflipvars(occ, x, y, vx, vy)
    # TODO: Figure out whether each pixel is in the given range
    # _before_ calling `fill` to reduce the number of edges
    img_inner ~ Map(Map(render_pixel_noflipvars))(
        fill(fill(occ, ImageSideLength()), ImageSideLength()),
        fill(fill(x, ImageSideLength()), ImageSideLength()),
        fill(fill(y, ImageSideLength()), ImageSideLength()),
        xs(),
        fill(1:ImageSideLength(), ImageSideLength())
    )
    return (img_inner,)
end

model_noflipvars = @DynamicModel(init_latent_model, step_latent_model, obs_model_noflipvars, 5)
@load_generated_functions
