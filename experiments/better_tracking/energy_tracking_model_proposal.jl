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
truncate(pvec, min, max, dom) = [
        (min ≤ i + first(dom) - 1 ≤ max ? p : 0.) for (i, p) in enumerate(pvec)
       ] |> normalize

### Domains for discrete values ###
Xs() = 1:20; Vels() = -3:3; Energies() = 1:10
Bools() = [true, false]
HOME() = 2 # home at the 10th x position

### Definitions of probability distributions ###
moving_away_from_home(vₜ₋₁, xₜ₋₁) = sign(vₜ₋₁) == sign(xₜ₋₁ - HOME())
prior_p_stop_tired(eₜ₋₁) = exp(-eₜ₋₁/5)
prior_p_stop_far(xₜ₋₁, vₜ₋₁) = moving_away_from_home(vₜ₋₁, xₜ₋₁) ?
                                    1-exp(-abs(xₜ₋₁-HOME())/5) : 0.
prop_p_stop_far(is_stopped, vₜ₋₁, xₜ₋₁) = !is_stopped ? 0. :
                       moving_away_from_home(vₜ₋₁, xₜ₋₁) ? 0.5 : 0.
prop_p_stop_tired(is_stopped, already_stopped, eₜ₋₁) =
    !is_stopped ? 0. : already_stopped ? prior_p_stop_tired(eₜ₋₁) : 0.6
expected_energy(eₜ₋₁, vₜ₋₁) = eₜ₋₁ + (abs(vₜ₋₁) > 0 ? -abs(vₜ₋₁) : 2.)
or(a, b) = a || b

### Model & Proposal ###
@gen (static) function step_model(xₜ₋₁, far, eₜ₋₁, vₜ₋₁, trd)
    stop_because_tired ~ bernoulli(prior_p_stop_tired(eₜ₋₁))
    stop_because_far ~ bernoulli(prior_p_stop_far(xₜ₋₁, vₜ₋₁))
    stop = or(stop_because_tired, stop_because_far)
    vₜ ~ LabeledCPT{Int}(
        [Bools(), Vels()],
        Vels(),
        ((stop, vₜ₋₁),) -> stop ?
                onehot(0, Vels()) : # if stopping, velocity = 0
                # else, w.p. 0.8, change velocity by 1 or 2
                maybe_one_or_two_off(vₜ₋₁, 0.8, Vels())
        )(stop, vₜ₋₁)
    xₜ ~ categorical(maybe_one_off(xₜ₋₁ + vₜ, 0.6, Xs()))
    eₜ ~ categorical(maybe_one_off(expected_energy(eₜ₋₁, vₜ₋₁), 0.5, Energies()))
    obsₜ ~ categorical(discretized_gaussian(xₜ, 2.0, Xs()))
    return obsₜ
end
@gen (static) function step_proposal(xₜ₋₁, vₜ₋₁, trd, far, eₜ₋₁, obsₜ)
    projection = min(max(first(Xs()), xₜ₋₁ + vₜ₋₁), last(Xs()))
    mean = (obsₜ + projection)/2
    xₜ ~ categorical(discretized_gaussian((Int ∘ floor)(mean), 1.5, Xs()))
    truncd_diff = min(last(Vels()), max(first(Vels()), xₜ - xₜ₋₁))
    vₜ ~ LabeledCPT{Int}(
        [Vels()],
        Vels(),
        ((diff,),) -> maybe_one_or_two_off(diff, 0.5, Vels())
    )(truncd_diff)
    stopped = vₜ == 0
    stop_because_far ~ bernoulli(prop_p_stop_far(stopped, vₜ₋₁, xₜ₋₁))
    stop_because_tired ~ bernoulli(prop_p_stop_tired(stopped, stop_because_far, eₜ₋₁))
    eₜ ~ categorical(maybe_one_off(expected_energy(eₜ₋₁, vₜ₋₁), .5, Energies()))
    return xₜ
end
@load_generated_functions()

args = (x0, v0, e0, obs1) = (10, 2, 8, 11)
step_proposal(x0, v0, false, false, e0, obs1)

# truncate(pvec, min, max) = [p for (i, p) in enumerate(pvec) if min ≤ i ≤ max] |> normalize

(xₜ₋₁, vₜ₋₁, trd, far, eₜ₋₁, obsₜ) = x0, v0, false, false, e0, obs1

# minvel = min(vₜ₋₁ - 2, 0)
# maxvel = max(vₜ₋₁ + 2, 0)
# xmin = xₜ₋₁ + minvel - 1
# xmax = xₜ₋₁ + maxvel + 1
# xₜ ~ categorical(
#     truncate(
#         discretized_gaussian(obsₜ, 2.0, Xs()),
#         xmin, xmax, Xs()
#     )
# )

# vₜ ~ LabeledCategorical(Vels())(truncate(
#     maybe_one_or_two_off(xₜ - xₜ₋₁, 0.5, Vels()),
#     minvel, maxvel, Vels()
# )

