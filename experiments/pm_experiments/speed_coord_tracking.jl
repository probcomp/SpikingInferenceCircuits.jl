# TODO: a version of this model where `speed` and `dir`
# control the motion, rather than a `vx` and `vy` which change independently

@gen (static) function initial_latent_model()
    xₜ ~ Cat(unif(Positions()))
    yₜ ~ Cat(unif(Positions()))
    speed ~ LCat(Dirs())(unif(Speeds()))
    dir ~ LCat(Dirs())(unif(Dirs()))

    return (xₜ, yₜ, speed, dir)
end
@gen (static) function step_latent_model(xₜ₋₁, yₜ₋₁, speedₜ₋₁, dirₜ₋₁)
    dirₜ ~ LCat(Dirs())(circular_truncated_gaussian(dirₜ₋₁, 2.0, Dirs()))
    speedₜ ~ LCat(Dirs())(truncated_gaussian(speedₜ₋₁, 2.0, Dirs()))
    
    vx = approximate_x_component(dirₜ, speedₜ)
    expected_new_x
    xₜ ~ Cat(truncated_gaussian())
end



@gen (static) function obs_model(xₜ, vxₜ, yₜ, vyₜ)
    xy = (xₜ, yₜ)
    obspos ~ LCat(Positions2d())(truncated_discretized_gaussian2d(xy, 3.0, Positions()))

    return obspos
end
