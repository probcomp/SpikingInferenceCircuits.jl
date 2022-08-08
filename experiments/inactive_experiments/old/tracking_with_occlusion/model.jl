includet("model_hyperparams.jl")
includet("model_utils.jl")

### initial & step model ###
@gen (static) function initial_latents(go::Nothing)
	occ ~ categorical(uniform(positions(let _=go; OccluderLength(); end)))
	x ~ categorical(uniform(positions(let _=go; SquareSideLength(); end)))
	y ~ categorical(uniform(positions(let _=go; SquareSideLength(); end)))
	vx ~ LabeledCategorical(Vels(), uniform(Vels()))(go)
	vy ~ LabeledCategorical(Vels(), uniform(Vels()))(go)

	return go
end
@load_generated_functions
println(initial_latents)

VStepDist = LabeledCPT{Int}([Vels()], Vels(), ((v,),) -> maybe_one_off(v, 0.4, Vels()))
@gen (static) function step(occ, x, y, vx, vy)
	occ ~ categorical(maybe_one_off(occ, 0.3, positions(OccluderLength())))
	x ~ categorical(truncated_discretized_gaussian(x + vx, 2.,
			positions(SquareSideLength()))
	)
	y ~ categorical(truncated_discretized_gaussian(y + vy, 2.,
		positions(SquareSideLength()))
	)

    vx ~ VStepDist(vx)
    vy ~ VStepDist(vy)

	return occ
end

### Obs model ###
## deterministic util functions
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
    got_photon ~ bernoulli(expect_pixel_on ? 0.9 : 0.1)
    return got_photon
end

@gen (static) function observation(occ, x, y, vx, vy)
    # TODO: Figure out whether each pixel is in the given range
    # _before_ calling `fill` to reduce the number of edges
    img ~ Map(Map(render_pixel))(
        fill(fill(occ, ImageSideLength()), ImageSideLength()),
        fill(fill(x, ImageSideLength()), ImageSideLength()),
        fill(fill(y, ImageSideLength()), ImageSideLength()),
        fill(1:ImageSideLength(), ImageSideLength()),
        [[pixx for _=1:ImageSideLength()] for pixx=1:ImageSideLength()]
    )

    return img
end

@gen (static) function obs_1d(occ, x)
    pixline ~ Map(render_pixel)(
        fill(occ, ImageSideLength()),
        fill(x, ImageSideLength()),
        fill(1, ImageSideLength()),
        1:ImageSideLength(),
        fill(1, ImageSideLength())
    )

    return pixline
end

println("Model file loaded.")