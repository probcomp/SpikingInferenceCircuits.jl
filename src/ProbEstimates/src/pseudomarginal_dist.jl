struct PMDistTrace{R} <: Gen.Trace
    gf::Gen.GenerativeFunction
    args::Tuple
    retval::R
end
Gen.get_args(tr::PMDistTrace) = tr.args
Gen.get_retval(tr::PMDistTrace) = tr.retval
Gen.get_choices(tr::PMDistTrace) = StaticChoiceMap((val=get_retval(tr),), (;))
Gen.get_score(tr::PMDistTrace) = -recip_prob_estimate(tr)
Gen.get_gen_fn(tr::PMDistTrace) = tr.gf

# Not totally sure this is right--
Gen.project(tr::CatTrace, ::EmptySelection) = 0.

struct PseudoMarginalizedDist{R} <: Gen.GenerativeFunction{R, PMDistTrace}
    model::Gen.GenerativeFunction{R}
    proposal::Gen.GenerativeFunction
    ret_trace_addr
    n_particles::Int
end
Base.:(==)

Gen.simulate(d::PseudoMarginalizedDist{R}, args) where {R} =
    PMDistTrace(d, args, get_retval(simulate(d.model, args)))
function Gen.generate(d::PseudoMarginalizedDist{R}, args::Tuple, cm::Union{Gen.ChoiceMap, Gen.EmptyChoiceMap}) where {R}
    tr = if isempty(cm)
        simulate(d, args)
    else
        @assert has_value(cm, :val)
        @assert length(collect(get_submaps_shallow(cm))) == 0
        @assert length(collect(get_values_shallow(cm))) == 1
        PMDistTrace(d, args, cm[:val])
    end
    return (tr, log_fwd_prob_estimate(tr))
end
function Gen.update(tr::PseudoMarginalizedDist, args::Tuple, ::Tuple, cm::Gen.ChoiceMap)
    if isempty(cm) && args == get_args(tr)
        return (tr, 0., NoChange(), EmptyChoiceMap())
    else
        error("Not expecting nontrivial `update` to be called on `PseudoMarginalizedDist`. args $(args == get_args(tr) ? "did not" : "did") change; cm = $cm")
    end
end

log_fwd_prob_estimate(tr::PMDistTrace) =
    n_propose_assess_cycles(get_gen_fn(tr), get_retval(tr), get_args(tr))
function log_recip_prob_estimate(tr::PMDistTrace)
    if weighttype == :noisy
        # If we're using noisy weights, we need to invert how we take probability estimates,
        # so that we use the reciprocal estimation method for the `assess` calls,
        # and the fwd estimation method for the `propose` calls.
        # This ensures that we end up outputting an unbiased reciprocal weight
        # from the whole pseudo-marginalization procedure.
        # (If we used recip weights for propose, and fwd for assess, as we usually would,
        # then our estimate of the score of the trace will be unbiased--but the reciprocal
        # of this estimate will be biased!)
        est = -n_propose_assess_cycles(
            get_gen_fn(tr), get_retval(tr), get_args(tr);
            call_before_propose=use_only_fwd_weights!,
            call_before_assess=use_only_recip_weights!
        )
        use_noisy_weights!()
        return est
    else
        return -n_propose_assess_cycles(get_gen_fn(tr), get_retval(tr), get_args(tr))
    end
end

function n_propose_assess_cycles(d, val, args;
    call_before_propose=(() -> nothing),
    call_before_assess=(() -> nothing)
)
    weight_sum = 0.
    for _=1:d.n_particles
        call_before_propose()
        proposed_choices, proposed_score = propose(d.proposal, (val, args...))
        call_before_assess()
        assessed_score, v1 = assess(
            d.model, args, proposed_choices
        )
        @assert v1 == val "val = $val, v1 = $v1"

        weight_sum += exp(assessed_score - proposed_score)
    end

    return log(weight_sum) - log(d.n_particles)
end