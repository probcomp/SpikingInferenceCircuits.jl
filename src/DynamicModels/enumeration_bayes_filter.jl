function unpack_assmts(latent_addr_to_domain)
    addrs = keys(latent_addr_to_domain)
    doms = values(latent_addr_to_domain)
    assmts = Iterators.product(doms...)
    to_choicemap(assmt) = choicemap(zip(addrs, assmt)...)

    return (addrs, assmts, Iterators.map(to_choicemap, assmts))
end
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

function enumeration_filter_step(step_model, obs_model, obs_choicemap, latent_addr_to_domain, prev_latent_weights, prev_latent_retvals)
    (_, _, assmt_choicemaps) = unpack_assmts(latent_addr_to_domain)

    first_prev_ret   , rest_prev_rets    = Iterators.peel(prev_latent_retvals)
    first_prev_weight, rest_prev_weights = Iterators.peel(prev_latent_weights)

    step_weights, retvals = get_weights_retvals(step_model, first_prev_ret, assmt_choicemaps)
    step_weights = step_weights .+ first_prev_weight

    for (prev_ret, prev_weight) in zip(rest_prev_rets, rest_prev_weights)
        weightterms = get_weights_retvals(step_model, prev_ret, assmt_choicemaps)[1] .+ prev_weight
        step_weights = log.(exp.(step_weights) + exp.(weightterms))
    end

    obs_weights = [generate(obs_model, latent_ret, obs_choicemap)[2] for latent_ret in retvals]

    return (step_weights + obs_weights, retvals)
end

struct EnumerationBayesFilter
    init_model
    step_model
    obs_model
    obs_choicemaps
    latent_addr_to_domain
end
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
    weights, rets = enumeration_filter_step(e.step_model, e.obs_model, obs_choicemap, e.latent_addr_to_domain, prev_weights, prev_rets)
    return (weights, (weights, rets, obs_iter_st))
end
enumeration_bayes_filter_from_groundtruth(dynamic_model_tr, init_model, step_model, obs_model, latent_addr_to_domain) =
    let (firstobs, restobs) = get_dynamic_model_obs(dynamic_model_tr)
        EnumerationBayesFilter(
            init_model, step_model, obs_model,
            Iterators.flatten(((firstobs,), restobs)) |> collect,
            latent_addr_to_domain
        )
    end

nest_all_addrs_at_val(e::EnumerationBayesFilter) = EnumerationBayesFilter(
    e.init_model, e.step_model, e.obs_model, e.obs_choicemaps,
    Dict((pairnest(addr, :val), dom) for (addr, dom) in pairs(e.latent_addr_to_domain))
)
pairnest(p::Pair, ending) = pairnest(p.first, pairnest(p.second, ending))
pairnest(v, ending) = v => ending