using Gen
using CPTs
using Distributions

### Distributions on probs, inverse probs, multiplying probs ###
K() = 20
# for multiplier:
timer_nspikes() = 30
timer_expected_time() = 50.

sample_prob_val(probs, val, dom) = sample_prob_val(probs, val - first(dom) + 1)
sample_prob_val(probs, val) = sample_prob_val(probs[val])
sample_prob_val(p) = (p, rand(Binomial(K(), p)))

sample_invprob_val(probs, val, dom) = sample_invprob_val(probs, val - first(dom) + 1)
sample_invprob_val(probs, val) = sample_invprob_val(probs[val])
function sample_invprob_val(p)
    time = rand(Erlang(K(), 1/p))
    nonselected_count = rand(Poisson((1 - p) * time))
    return (1/p, K() + nonselected_count)
end

function sample_product_val(counts...)
    timerrate = timer_nspikes() / timer_expected_time()
    time = rand(Erlang(timer_nspikes(), 1/timerrate))
    multrate = K() * prod(counts) / (K()^length(counts)) / timer_expected_time()
    return rand(Poisson(multrate * time))
end


### Use these on the proposal & model ###
function get_proposal_weights(ch, (xₜ₋₁, vₜ₋₁, trd, far, eₜ₋₁, obsₜ))
    x_invprob = sample_invprob_val(
        discretized_gaussian(obsₜ, 1.5, Xs()), ch[:xₜ]
    )
    # println("""
    # diff = $(ch[:xₜ] - xₜ₋₁)
    # Vel = $( ch[:vₜ]);
    # prob = $(maybe_one_or_two_off(ch[:xₜ] - xₜ₋₁, 0.5, Vels())[ch[:vₜ] - first(Vels()) + 1])
    # """)
    truncd_diff = min(last(Vels()), max(first(Vels()), ch[:xₜ] - xₜ₋₁))
    v_invprob = sample_invprob_val(
        maybe_one_or_two_off(truncd_diff, 0.5, Vels()), ch[:vₜ], Vels()
    )

    stopped = ch[:vₜ] == 0
    pfar = prop_p_stop_far(stopped, vₜ₋₁, xₜ₋₁)
    far_invprob = sample_invprob_val(
        [pfar, 1 - pfar], ch[:stop_because_far] ? 1 : 2
    )
    ptired = prop_p_stop_tired(stopped, ch[:stop_because_far], eₜ₋₁)
    tired_invprob = sample_invprob_val(
        [ptired, 1 - ptired], ch[:stop_because_tired] ? 1 : 2
    )
    
    e_invprob = sample_invprob_val(
        maybe_one_off(expected_energy(eₜ₋₁, ch[:vₜ]), .5, Energies()), ch[:eₜ]
    )

    return (x_invprob, v_invprob, far_invprob, tired_invprob, e_invprob)
end
function get_model_weights(ch, (xₜ₋₁, vₜ₋₁, trd, far, eₜ₋₁, obsₜ))
    p_trd = prior_p_stop_tired(eₜ₋₁)
    trd_prob = sample_prob_val([p_trd, 1 - p_trd], ch[:stop_because_tired] ? 1 : 2)

    p_far = prior_p_stop_far(xₜ₋₁, vₜ₋₁)
    far_prob = sample_prob_val([p_far, 1 - p_far], ch[:stop_because_far] ? 1 : 2)
    
    v_prob = if ch[:stop_because_far] || ch[:stop_because_tired]
        sample_prob_val(onehot(0, Vels()), ch[:vₜ], Vels())
    else
        sample_prob_val(maybe_one_or_two_off(vₜ₋₁, 0.8, Vels()), ch[:vₜ], Vels())
    end

    x_prob = sample_prob_val(maybe_one_off(xₜ₋₁ + ch[:vₜ], 0.6, Xs()), ch[:xₜ])
    e_prob = sample_prob_val(maybe_one_off(expected_energy(eₜ₋₁, ch[:vₜ]), 0.5, Energies()), ch[:eₜ])
    obs_prob = sample_prob_val(discretized_gaussian(ch[:xₜ], 2.0, Xs()), obsₜ)

    return (trd_prob, far_prob, v_prob, x_prob, e_prob, obs_prob)
end

function get_importance_weight_sample(x0, v0, e0, obs1)
    propargs = (x0, v0, false, false, e0, obs1)
    prop_choices, propweight, _ = propose(step_proposal, (x0, v0, false, false, e0, obs1))
    assessweight, _ = assess(step_model, (x0, false, e0, v0, false), merge(choicemap((:obsₜ, obs1)), prop_choices))
    
    prop_weights = get_proposal_weights(prop_choices, propargs)
    true_p_weights = map(first, prop_weights)
    p_counts = map(last, prop_weights)

    model_weights = get_model_weights(prop_choices, propargs)
    true_m_weights = map(first, model_weights)
    m_counts = map(last, model_weights)

    # println(true_m_weights)
    tr, _ = generate(step_model, (x0, false, e0, v0, false), merge(prop_choices, choicemap((:obsₜ, obs1))))
    # println(Tuple(
    #     exp(project(tr, select(varname)))
    #     for varname in (:stop_because_tired, :stop_because_far, :vₜ, :xₜ, :eₜ, :obsₜ)
    # ))
    # display(get_choices(tr))
    # display(prop_choices)
    
    @assert isapprox(prod(true_p_weights), exp(-propweight))
    @assert isapprox(prod(true_m_weights), exp(assessweight)) "$(prod(true_m_weights)) | $(exp(assessweight))"


    true_val = exp(assessweight - propweight)
    other_true_val = prod([true_p_weights..., true_m_weights...])
    @assert isapprox(true_val, other_true_val) "gen: $true_val ; manual: $other_true_val"

    multval = sample_product_val(p_counts..., m_counts...)

    return (true_val, multval)
end

function get_importance_weight_samples(N, args...)
    return [
        get_importance_weight_sample(args...)
        for _=1:N
    ]
end

samples = get_importance_weight_samples(10000, x0, v0, e0, obs1)

trueweights = map(first, samples)
counts = map(last, samples)

unique_trues = sort!(unique(trueweights))
corresponding_counts = [
    [cnt for (tru, cnt) in zip(trueweights, counts) if tru == t]
    for t in unique_trues
]
avg_counts = [sum(cnts)/length(cnts) for cnts in corresponding_counts]
vars = [
    sum((cnt^2 - avg^2) for cnt in cnts)/length(cnts)
    for (cnts, avg) in zip(corresponding_counts, avg_counts)
]


plt = scatter(unique_trues, avg_counts ./K(), color=:black)
# sub1_inds = findall(x -> x < 0.4, trueweights)
# scatter!(trueweights[sub1_inds], counts[sub1_inds]./K())

errorbars!(unique_trues, avg_counts ./ K(), sqrt.(vars)./K(), color=:red)
plt
# ax = scatter(trueweights, counts./K())
lines!([0., 3.5], [0., 3.5])
plt
# scatter!(.02:.02:3, .02:.02:3)
# ax

plt
# lines!([0., 1.1], [0., 1.1])
# # scatter!(.02:.02:3, .02:.02:3)
# ax

## TODO: scatter plot of these