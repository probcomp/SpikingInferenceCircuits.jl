function unpack_assmts(latent_addr_to_domain, n_to_skip=0)
    addrs  = Iterators.drop(keys(latent_addr_to_domain), n_to_skip)
    doms   = Iterators.drop(values(latent_addr_to_domain), n_to_skip)
    assmts = Iterators.product(doms...)
    to_choicemap(assmt) = choicemap(zip(addrs, assmt)...)

    return (addrs, assmts, Iterators.map(to_choicemap, assmts))
end

# Get the weights and retvals for each constraint on the model.  `detaddrs`
# should give the addresses of variables in the model which are deterministic given
# the constrained addrs, and so should end up being set to the right value automatically.
# This function returns the weight for each trace, as well as the retval, and the values
# for the deterministic variables.
function get_weights_retvals_detvals(model, args, iterator_over_constraints, detaddrs)
    trs_weights = [generate(model, args, c) for c in iterator_over_constraints]
    weights = [w for (_, w) in trs_weights]
    retvals = [get_retval(tr) for (tr, _) in trs_weights]
    detvals = [
                [tr[detaddr] for detaddr in detaddrs]
                for (tr, _) in trs_weights
            ]

    return (weights, detvals, retvals)
end

# Could use `get_weights_retvals_detvals` for this. I kept this fn separate since I figure it may be
# more performant this way since we never extract the detvals.
function get_weights_retvals(model, args, iterator_over_constraints)
    trs_weights = [generate(model, args, c) for c in iterator_over_constraints]
    weights = [w for (_, w) in trs_weights]
    retvals = [get_retval(tr) for (tr, _) in trs_weights]

    return (weights, retvals)
end

function enumeration_filter_init(init_model, obs_model, obs_choicemap, latent_addr_to_domain)
    (_, _, assmt_choicemaps) = unpack_assmts(latent_addr_to_domain)
    (init_weights, retvals) = get_weights_retvals(init_model, (), assmt_choicemaps)
    obs_weights = [generate(obs_model, latent_ret, obs_choicemap)[2] for latent_ret in retvals]

    return (init_weights + obs_weights, retvals)
end

function enumeration_filter_step(
    step_model::GenerativeFunction{T}, obs_model, obs_choicemap,
    latent_addr_to_domain, prev_latent_weights, prev_latent_retvals,
    num_determ_addrs
) where {T}
    (b, c, assmt_choicemaps) = unpack_assmts(latent_addr_to_domain, num_determ_addrs)

    # convert the values of the deterministic addresses into the indices in the array for those variables
    to_idxs(vals) = (v - first(dom) + 1 for (v, dom) in zip(vals, values(latent_addr_to_domain)))

    step_weights = [-Inf for _ in Iterators.product(values(latent_addr_to_domain)...)]
    retvals      = Array{Union{T, Nothing}}(nothing, map(length, values(latent_addr_to_domain))...)

    for (prev_ret, prev_weight) in zip(prev_latent_retvals, prev_latent_weights)
        if prev_weight > -Inf # skip this if the parents have 0 probability!
            # get the weights, retvals, and values for deterministic variables
            (weights, determ_vals, rets) = get_weights_retvals_detvals(
                step_model, prev_ret, assmt_choicemaps,
                Iterators.take(keys(latent_addr_to_domain), num_determ_addrs)
            )
            weightterms = weights .+ prev_weight

            # Set the retval and weight for each of these.  To get the index, we need to look
            # at the values which occurred for the deterministic variables.
            for (restidx, weight, ret, det) in zip(keys(weightterms), weightterms, rets, determ_vals)
                idx = (to_idxs(det)..., restidx)
                retvals[idx...] = ret
                step_weights[idx...] = log(exp(step_weights[idx...]) + exp(weight))
            end
        end
    end

    obs_weights = [isnothing(latent_ret) ? -Inf : generate(obs_model, latent_ret, obs_choicemap)[2] for latent_ret in retvals]

    return (step_weights + obs_weights, retvals)
end

struct EnumerationBayesFilter
    init_model
    step_model
    obs_model
    obs_choicemaps
    latent_addr_to_domain
    n_determ_addrs_in_step::Int
end
EnumerationBayesFilter(i, s, om, oc, l) = EnumerationBayesFilter(i, s, om, oc, l, 0)
Base.length(e::EnumerationBayesFilter) = length(e.obs_choicemaps)
function Base.iterate(e::EnumerationBayesFilter)
    i = iterate(e.obs_choicemaps)
    i === nothing && return nothing;
    obs_choicemap, obs_iter_st = i

    weights, rets = enumeration_filter_init(e.init_model, e.obs_model, obs_choicemap, e.latent_addr_to_domain)
    return (weights, (weights, rets, obs_iter_st))
end
function Base.iterate(e::EnumerationBayesFilter, (prev_weights, prev_rets, prev_obs_iter_st))
    i = iterate(e.obs_choicemaps, prev_obs_iter_st)
    i === nothing && return nothing;
    obs_choicemap, obs_iter_st = i
    weights, rets = enumeration_filter_step(e.step_model, e.obs_model, obs_choicemap, e.latent_addr_to_domain, prev_weights, prev_rets, e.n_determ_addrs_in_step)
    return (weights, (weights, rets, obs_iter_st))
end
enumeration_bayes_filter_from_groundtruth(dynamic_model_tr, init_model, step_model, obs_model, latent_addr_to_domain, n_determ_addrs_in_step=0) =
    let (firstobs, restobs) = get_dynamic_model_obs(dynamic_model_tr)
        EnumerationBayesFilter(
            init_model, step_model, obs_model,
            Iterators.flatten(((firstobs,), restobs)) |> collect,
            latent_addr_to_domain,
            n_determ_addrs_in_step
        )
    end

nest_all_addrs_at_val(e::EnumerationBayesFilter) = EnumerationBayesFilter(
    e.init_model, e.step_model, e.obs_model, e.obs_choicemaps,
    Dict((pairnest(addr, :val), dom) for (addr, dom) in pairs(e.latent_addr_to_domain)),
    e.n_determ_addrs_in_step
)
pairnest(p::Pair, ending) = pairnest(p.first, pairnest(p.second, ending))
pairnest(v, ending) = v => ending