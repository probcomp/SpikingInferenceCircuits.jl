### initial & step model ###
@gen (static) function initial_latents()
	occ ~ categorical(uniform(positions(OccluderLength())))
	x ~ categorical(uniform(positions(SquareSideLength())))
	y ~ categorical(uniform(positions(SquareSideLength())))
	vx ~ VelCat(uniform(Vels()))
	vy ~ VelCat(uniform(Vels()))
	return (occ, x, y, vx, vy)
end

@gen (static) function step(occ, x, y, vx, vy)
	occ ~ categorical(maybe_one_off(occ, 0.3, positions(OccluderLength())))
	x ~ categorical(truncated_discretized_gaussian(x + vx, 2.,
			positions(SquareSideLength()))
	)
	y ~ categorical(truncated_discretized_gaussian(y + vy, 2.,
		positions(SquareSideLength()))
	)
	vx ~ VelCat(maybe_one_off(vx, 0.4, Vels()))
	vy ~ VelCat(maybe_one_off(vy, 0.4, Vels()))
	return (occ, x, y, vx, vy)
end

### Obs model ###
p_got_photon(occ, sqx, sqy, x, y) = (
		is_occluded(occ, x, y) || is_in_square(sqx, sqy, x, y)
	) ? 1 - p_flip() : p_flip()

@gen (static) function render_pixel(a1, a2)
    got_photon ~ BoolCat(
        let p = p_got_photon(a1..., a2...); [p, 1-p]; end
    )
    return got_photon
end

@gen (static) function observation(occ, x, y, vx, vy)
    arg1, arg2 = Map2Dargs(
        fill((occ, x, y), (ImageSideLength(), ImageSideLength())),
        [(x, y) for x=1:ImageSideLength(), y=1:ImageSideLength()]
    )
    img_inner ~ Map2D(render_pixel)(arg1, arg2)
    return img_inner
end
