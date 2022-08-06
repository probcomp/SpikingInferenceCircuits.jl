using Gen

onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]
# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)
maybe_one_or_two_off(idx, prob, dom) = 
    (1 - prob) * onehot(idx, dom) +
    prob/3 * onehot(idx - 1, dom) +
    prob/3 * onehot(idx + 1, dom) +
    prob/6 * onehot(idx - 2, dom) +
    prob/6 * onehot(idx + 2, dom)

normalize(v) = v / sum(v)
discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
])

Xs = collect(1:40)
HOME = 20
Vels = collect(-4:4)
Energies = collect(1:30)
@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

struct PseudoMarginalizedDist{R} <: Gen.Distribution{R}
    model::Gen.GenerativeFunction{R}
    proposal::Gen.GenerativeFunction
    ret_trace_addr
    n_particles::Int
end
Gen.random(d::PseudoMarginalizedDist, args...) =
    get_retval(simulate(d.model, args))
function Gen.logpdf(d::PseudoMarginalizedDist, val, args...)
    weight_sum = 0.
    for _=1:d.n_particles
        proposed_choices, proposed_score = propose(d.proposal, (val, args...))
        assessed_score, v1 = assess(
            d.model, args,
            merge(choicemap((d.ret_trace_addr, val)), proposed_choices)
        )
        @assert v1 == val "val = $val, v1 = $v1"

        weight_sum += exp(assessed_score - proposed_score)
    end

    return log(weight_sum) - log(d.n_particles)
end

moving_away_from_home(x_tminus1, v_tminus1) = sign(v_tminus1) == sign(x_tminus1 - HOME)
dist_from_home(x_tminus1) = abs(x_tminus1 - HOME)
τ_far() = 10
@gen function vel_model(e_tminus1, v_tminus1, x_tminus1)
    stop_bc_tired = {:stop_tired} ~ bernoulli(exp(-e_tminus1))
    # 1 away, 4.8% chance of stopping. 10 away, 40%
    
    stop_bc_far_from_home = {:stop_far} ~ bernoulli(
        moving_away_from_home(x_tminus1, v_tminus1) ?
            1-exp(-dist_from_home(x_tminus1)/τ_far()) :
            0.
        )

    stop = stop_bc_tired || stop_bc_far_from_home

    v = {:v} ~ LabeledCategorical(Vels, 
        stop ? onehot(0, Vels) :
            maybe_one_or_two_off(v_tminus1, .8, Vels))

    return v
end

# need to marginalize out these two variables so we have to consider what we already know
# would really like to sum over all possible values, but instead are going to sample tons of possibilities, 
# which basically proposes the most important values contributing to the sum 
@gen function vel_auxiliary_proposal(v, e_tminus1, v_tminus1, x_tminus1)
    if v != 0
        {:stop_tired} ~ bernoulli(0)
        {:stop_far} ~ bernoulli(0)
        return
    end

    # else, v = 0

    # TODO: stop hardcoding these probability coding functions!!!
    # ie. have `p_stopped_tired`, `p_stopped_far`, etc., 
    p_stopped_tired = exp(-e_tminus1)
    p_stopped_far = moving_away_from_home(x_tminus1, v_tminus1) ? 1-exp(-dist_from_home(x_tminus1)/τ_far()) : 0.

    p_neither_stop_true = (1 - p_stopped_tired)*(1 - p_stopped_far)
    p_stopvar_true = 1 - p_neither_stop_true

    # p (v = 0   ;  did not stop, v_tminus1)   (Prob v = 0 from being 1 or 2 off)
    # TODO: remove hardcoded prob 0.5
    p_off = maybe_one_or_two_off(0, 0.5, Vels)[v_tminus1 - first(Vels) + 1]

    approx_marginal_p_stop = p_stopvar_true / (p_stopvar_true + p_off)
    approx_p_tired_given_stop = p_stopped_tired / (p_stopped_tired + p_stopped_far)

    stop_tired = {:stop_tired} ~ bernoulli(approx_marginal_p_stop * approx_p_tired_given_stop)
    {:stop_far} ~ bernoulli(stop_tired ?
                        p_stopped_far : 
                        approx_marginal_p_stop
                    )
end
vel_dist = PseudoMarginalizedDist(
    vel_model,
    vel_auxiliary_proposal,
    :v,
    10 # TODO: tune NParticles
)