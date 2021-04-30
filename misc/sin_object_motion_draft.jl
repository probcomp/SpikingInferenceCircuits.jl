# Work on recreating the readme example model from
# https://github.com/probcomp/GenParticleFilters.jl
# as a discrete-variable model we can compile.

const LOOPSIZE = 8
YDOMAIN = -2:0.25:2
is_in_domain(y, dom) = first(dom) ≤ y ≤ last(dom)

onehot(x, l) = [i == x ? 1 : 0 for i=1:l]
truncate(v, l) = v < 1 ? 1 : (v > l : l : v)
gridded_step_probs(idx, cont_change, domain) =
    let leftsquare_delta = (Int ∘ div)(cont_change, domain.step),
        rightprob = mod(cont_change, domain.step) / domain.step,
        vec = v -> onehot(truncate(v, length(domain)), length(domain))
           ((1 - rightprob) * vec(idx + leftsquare_delta) 
                            +
                  rightprob * vec(idx + leftsquare_delta + 1))
    end

@gen (static) function object_motion_step(yₜ₋₁, ϕₜ₋₁, movingₜ₋₁)
    ϕₜ      = mod(ϕₜ₋₁ + 1, LOOPSIZE)
    movingₜ ~ bernoulli(movingₜ₋₁ ? 0.75 : 0.25)

    # vel = moving ? sin(2pi * ϕₜ / LOOPSIZE) : 0.0
    # Using Switch, this is written as:
    vel ~ Switch(
        Determ(x -> sin(2pi * x / LOOPSIZE)),
        Determ(x -> 0)
    )(movingₜ ? 1 : 2, ϕₜ)

    yₜ ~ categorical(gridded_step_probs(yₜ₋₁, vel, YDOMAIN))
    obs ~ add_step_noise(yₜ)

    return obs
end
# TODO: Determ

# @gen function object_motion_step(yₜ₋₁, ϕₜ₋₁, movingₜ₋₁)
#     ϕₜ      = mod(ϕₜ₋₁ + 1, LOOPSIZE)
#     movingₜ ~ bernoulli(movingₜ₋₁ ? 0.75 : 0.25)
#     vel = moving ? sin(2pi * ϕₜ / LOOPSIZE) : 0.0
#     sum = yₜ₋₁ + vel
#     yₜ = is_inside_yrange(sum) ? sum : yₜ₋₁
#     return yₜ
# end
