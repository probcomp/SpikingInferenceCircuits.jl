"""
Dict from `(compname, spike::Sim.Spike)` to vector of all times when a Spike of that type occurred.

(A `Spike` is either an `InputSpike` or `OutputSpike`, and specifies an input/output port (`spike.name`).)
"""
function spiketrain_dict(events)
    d = Dict()
    for (t, c, e) in events
        push!(get!(d, (c, e), []), t)
    end
    return d
end

### Get Sample, Recip Score, Score outputs ###
smcaddr(t) = t == 0 ? :initial_step : (:subsequent_steps => :smcstep)

sample_wta_addr(particle, var, t)   = Circuits.nest(smcaddr(t), :particles => particle => :propose        => :sub_gen_fns => var => :component => :sample => :wta)
recip_score_addr(particle, var, t)  = Circuits.nest(smcaddr(t), :particles => particle => :propose        => :sub_gen_fns => var => :component => :sample => :counter)
score_latent_addr(particle, var, t) = Circuits.nest(smcaddr(t), :particles => particle => :assess_latents => :sub_gen_fns => var => :component => :score)
score_obs_addr(particle, var, t)    = Circuits.nest(smcaddr(t), :particles => particle => :assess_obs     => :sub_gen_fns => var => :component => :score)

function find_varname_for_addr(events, addr, t, in_proposal)
    h = hierarchy_lookup(get_name_hierarchy(events), smcaddr(t))
    circuit_names = in_proposal ? (:propose,) : (:assess_latents, :assess_obs)
    possible_varnames = Iterators.flatten(
        keys(h[:particles][1][circuit_name][:sub_gen_fns])
        for circuit_name in circuit_names
    )

    for varname in possible_varnames
        if varname isa Symbol && contains(String(varname), String(addr))
            return varname
        end
    end
    return nothing
end

function events_in_timerange(events, first_evt_time, first_post_time_event)
    get_time((t, c, e)) = t
    first_event_idx = searchsortedfirst(events, (first_evt_time, nothing, nothing); by=get_time)
    last_evt_idx = searchsortedfirst(events, (first_post_time_event, nothing, nothing); by=get_time) - 1
    return @view(events[first_event_idx:last_evt_idx])
end

function get_whichever_key_exists(d, keys; default=nothing)
    if isempty(keys)
        return default
    else
        (fst, rst) = Iterators.peel(keys)
        if haskey(d, fst)
            @assert !any(haskey(d, k) for k in rst)
            return d[fst]
        else
            return get_whichever_key_exists(d, rst; default)
        end
    end
end

remove_nothing_values(d) = Dict(k => v for (k, v) in d if !isnothing(v))

function get_trains_for_step(
    events,
    relevant_addr_fns, particle, var_addrs,
    relevant_spikenames, # The names of the output ports we care about (for any of the sub-circuits we're getting spiketrains for)
    t, inter_obs_interval
)
    # only filter on the events in the right range, to speed things up
    events_in_range = events_in_timerange(events, inter_obs_interval * t, inter_obs_interval * (t + 1))
    
    # get the var name symbols for the addrs we care about
    propose_vars = map(a -> find_varname_for_addr(events_in_range, a, t, true), var_addrs)
    println("Propose var names: $propose_vars.")
    assess_vars = map(a -> find_varname_for_addr(events_in_range, a, t, false), var_addrs)
    println("Assess var names: $assess_vars.")

    addr_to_varnames = Dict(a => [prop, ass] for (a, prop, ass) in zip(var_addrs, propose_vars, assess_vars))

    varaddr_fn_to_circuit_addrs = Dict(
        (varaddr, relevant_addr_fn) => [relevant_addr_fn(particle, var, t) for var in varnames]
        for relevant_addr_fn in relevant_addr_fns for (varaddr, varnames) in addr_to_varnames
    )
    all_circuit_addrs = Set(Iterators.flatten(values(varaddr_fn_to_circuit_addrs)))

    relevant_evts = filter(((t, c, e),) -> c in all_circuit_addrs && e isa SpikingCircuits.SpikingSimulator.OutputSpike, events_in_range)
    println("Events filtered.")
    
    circuit_addr_dict = spiketrain_dict(relevant_evts)

    return Dict(
        (key, spike) => get_whichever_key_exists(circuit_addr_dict, (addr, spike) for addr in circuit_addrs)
        for (key, circuit_addrs) in varaddr_fn_to_circuit_addrs
            for spike in (SpikingSimulator.OutputSpike(name) for name in relevant_spikenames)
    ) |> remove_nothing_values
end

get_sample_score_recip_trains(events, particle, var_addrs, max_domain_size, t, inter_obs_interval) =
    get_trains_for_step(
        events,
        (sample_wta_addr, recip_score_addr, score_latent_addr, score_obs_addr),
        particle, var_addrs,
        [(1:max_domain_size)..., :count => :count, :count => :ind],
        t, inter_obs_interval
    )

sample_label_keys(varaddr, domain) = (["$varaddr=$v" for v in domain], [i for i=1:length(domain)])
score_label_keys(varaddr, domain) = (["P($varaddr) count", "P($varaddr) count ready"], [:count => :count, :count => :ind])
recip_score_label_keys(varaddr, domain) = (["1/Q($varaddr) count", "1/Q($varaddr) count ready"], [:count => :count, :count => :ind])

"""
Returns a list of (label, spiketrain) pairs.
"""
function get_sample_score_recip_labels_trains(events, particle, var_addr_domain_pairs, t, inter_obs_interval)
    d = get_sample_score_recip_trains(
        events, particle, map(first, var_addr_domain_pairs),
        maximum(length(d) for (_, d) in var_addr_domain_pairs),
        t, inter_obs_interval
    )

    return [
        (label, get_whichever_key_exists(d, [((addr, fn), SpikingSimulator.OutputSpike(key)) for fn in fns]; default=[]))
        for (addr, dom) in var_addr_domain_pairs
            for ((labels, keys), fns) in zip(
                [sample_label_keys(addr, dom), recip_score_label_keys(addr, dom), score_label_keys(addr, dom)],
                [[sample_wta_addr], [recip_score_addr], [score_latent_addr, score_obs_addr]]
            )
                for (label, key) in zip(labels, keys)
    ]
end

# TODO: issue: the forward-scores aren't even ending up in the dictionary!