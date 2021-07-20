# We override this method in `ProbEstimates` to simulate the noise in NeuralGen-Spiking.
function mh_p_accept(new_weight, bwd_weight, fwd_weight, old_tr_weight; update_weight)
    @assert isapprox(update_weight, new_weight - old_tr_weight)
    return exp(new_weight + bwd_weight - fwd_weight - old_tr_weight)
end

function last_timestep_mh(
    trace, proposal::GenerativeFunction, proposal_args::Tuple;
    check=false, observations=EmptyChoiceMap())
    # TODO add a round trip check
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)

    T = get_args(trace)[1]
    new_tr_constraints = merge(fwd_choices, get_selected(get_choices(trace), select(obs_addr(T))))
    old_tr_constraints = get_selected(get_choices(trace), select(addr_for_timestep(T)))
    if T > 0
        trace_before_timestep, _, _, _ = update(trace, (T - 1,), (UnknownChange(),), EmptyChoiceMap())

        new_trace, new_weight, _, _    = update(trace_before_timestep, (T,), (UnknownChange(),), new_tr_constraints)
        old_trace, old_tr_weight, _, _ = update(trace_before_timestep, (T,), (UnknownChange(),), old_tr_constraints)
        @assert get_choices(old_trace) == get_choices(trace)
    else
        new_trace, new_weight    = generate(get_gen_fn(trace), get_args(trace), new_tr_constraints)
        old_trace, old_tr_weight = generate(get_gen_fn(trace), get_args(trace), get_choices(trace))
        @assert get_choices(old_trace) == get_choices(trace)
    end
    _new_trace, update_weight, _, discard = update(trace, get_args(trace), (NoChange(),), fwd_choices)
    if get_choices(_new_trace) != get_choices(new_trace)
        @error("New trace != new trace generated other way!")
        println("_new_trace:")
        display(get_choices(_new_trace))
        println("new_trace:")
        display(get_choices(new_trace))
    end

    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)

    check && check_observations(get_choices(new_trace), observations)
    
    if rand() < mh_p_accept(new_weight, bwd_weight, fwd_weight, old_tr_weight; update_weight)
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end