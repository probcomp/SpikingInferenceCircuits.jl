### Object motion model + inference code for possible figure.

# UTILS:
# onehot vector for `x` with length `length(dom)`,
# with `x` truncated to domain
onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 0 : 1 for i in dom]

# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)

# MODEL:
XDOMAIN = 1:20 # = [1, 2, ..., 20]

# Annotate inputs with their domains
@gen (static) function object_motion_step(xₜ₋₁::XDOMAIN,
                                      velₜ₋₁::[-1, 0, 1])
    # change velocity with small probability
    velₜ ~ LabeledCategorical(-1, 0, 1)(
        velₜ₋₁ == -1 ? [.8, .2, .0] :
        velₜ₋₁ ==  0 ? [.2, .6, .2] :
                       [.0, .2, .8]
    )
    xₜ   ~ categorical(maybe_one_off(xₜ₋₁ + velₜ, 0.1, XDOMAIN))
    obsₜ ~ categorical(maybe_one_off(xₜ, 0.3, XDOMAIN))
    return obsₜ
end

## Inference Program

# Bottom-up proposal for `xₜ` and `velₜ`, which _ignores_
# the prior probability of changing velocity from `velₜ₋₁`.
@gen (static) function step_proposal(xₜ₋₁::XDOMAIN, velₜ₋₁::[-1, 0, 1], obsₜ::XDOMAIN)
    xₜ ~ categorical(maybe_one_off(obsₜ, 0.3, XDOMAIN))
    velₜ ~ categorical(maybe_one_off(xₜ - xₜ₋₁, 0.1, [-1, 0, 1]))
end

inference_circuit = SMC_Circuit( object_motion_step, step_proposal; num_particles=3, resample=true )

# SMC algorithm
N = 4 # 4 particles
@particle function SMC_Particle(xₜ₋₁, velₜ₋₁, obsₜ)
    proposed, one_over_q = PROPOSE(proposal, xₜ₋₁, velₜ₋₁, obsₜ)
                       p  = ASSESS(model, proposed, obsₜ)
    return (proposed, p * one_over_q)
end
@particles function SMC_Step(particle_params)
    particle_outputs = [ SMC_Particle(particle_params[i]..., obsₜ) for i=1:N ]
    return Resample(N, particle_outputs)
end

# This could be declared in a standard library:
@particles function Resample(N, particles)
    proposed_traces, weights = unzip(particles)
    resampled_indices = [normalize_and_sample(weights) for i=1:N]
    return [ proposed_traces[i] for i in resampled_indices ]
end

# The full inference loop
@recurrent_circuit function InferenceCircuit(RECURRENT_OUTPUT, obsₜ)
    SMC_Step(SYNCRONIZE(obsₜ, RECURRENT_OUTPUT))
end

# Idea is:
# - An `@particle` circuit returns a `(trace, weight)` pair
# - An `@particles` circuit returns a vector of `(trace, weight)` pairs

### Draft 2 ###
maybe_one_off(i, p, d) = ((1 - p) * onehot(i, d) +
    p/2 * onehot(i - 1, d) + p/2 * onehot(i + 1, d) )
XDOMAIN = 1:20 # = [1, 2, ..., 20]

@gen (static) function object_motion_step(
        xₜ₋₁::XDOMAIN, velₜ₋₁::[-1, 0, 1]   )
    # change velocity with small probability
    velₜ ~ LabeledCategorical(-1, 0, 1)(
        velₜ₋₁ == -1 ? [.8, .2, .0] :
        velₜ₋₁ ==  0 ? [.2, .6, .2] :
                       [.0, .2, .8] )
    # w.p. 0.9, xₜ = xₜ₋₁ + velₜ ; otherwise it will be 1 off
    xₜ ~ categorical(maybe_one_off(xₜ₋₁ + velₜ, 0.1, XDOMAIN))
    # w.p. 0.7, obsₜ = xₜ ; otherwise it will be 1 off
    obsₜ ~ categorical(maybe_one_off(xₜ, 0.3, XDOMAIN))
    return obsₜ
end
