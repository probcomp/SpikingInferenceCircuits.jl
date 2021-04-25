# Working on recreating the readme example model from
# https://github.com/probcomp/GenParticleFilters.jl
# as a discrete-variable model we can compile.

const LOOPSIZE = 8
const MINY = -2
const MAXY = 2
is_inside_yrange(y) = MINY < y < MAXY

@gen (static) function object_motion_step(yₜ₋₁, ϕₜ₋₁, movingₜ₋₁)
    ϕₜ      = mod(ϕₜ₋₁ + 1, LOOPSIZE)
    movingₜ ~ bernoulli(movingₜ₋₁ ? 0.75 : 0.25)

    # vel = moving ? sin(2pi * ϕₜ / LOOPSIZE) : 0.0
    # Using Switch, this is written as:
    vel ~ Switch(
        Determ(x -> sin(2pi * x / LOOPSIZE)),
        Determ(x -> 0)
    )(movingₜ ? 1 : 2, ϕₜ)

    sum = yₜ₋₁ + vel
    yₜ = is_inside_yrange(sum) ? sum : yₜ₋₁
    return yₜ
end
# TODO: determ

# @gen function object_motion_step(yₜ₋₁, ϕₜ₋₁, movingₜ₋₁)
#     ϕₜ      = mod(ϕₜ₋₁ + 1, LOOPSIZE)
#     movingₜ ~ bernoulli(movingₜ₋₁ ? 0.75 : 0.25)
#     vel = moving ? sin(2pi * ϕₜ / LOOPSIZE) : 0.0
#     sum = yₜ₋₁ + vel
#     yₜ = is_inside_yrange(sum) ? sum : yₜ₋₁
#     return yₜ
# end