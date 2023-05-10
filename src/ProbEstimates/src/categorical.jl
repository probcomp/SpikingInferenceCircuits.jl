mutable struct CatTrace{T} <: Gen.Trace
    gf::Gen.GenerativeFunction{T}
    probs::Vector{Float64}
    idx::Int
    fwd_score::Union{Nothing, Float64}
    recip_score::Union{Nothing, Float64}
    proposal_probs::Union{Nothing, Vector{Float64}}
end
CatTrace(gf, probs, idx) = CatTrace(gf, probs, idx, nothing, nothing, nothing)

Gen.get_args(tr::CatTrace) = (tr.probs,)
Gen.get_retval(tr::CatTrace) = idx_to_label(tr.gf, tr.idx)
Gen.get_choices(tr::CatTrace) = StaticChoiceMap((val=get_retval(tr), recip_score=tr.recip_score, fwd_score=tr.fwd_score, proposal_probs=tr.proposal_probs), (;))
function Gen.get_score(tr::CatTrace)
    -log(recip_prob_estimate(tr))
end
Gen.get_gen_fn(tr::CatTrace) = tr.gf

# I don't think this is right when using the noise model
function Gen.project(tr::CatTrace, s::Selection)
    if :val in s
        return log(tr.probs[tr.idx])
    else
        return 0.
    end
end

struct LCat{T} <: Gen.GenerativeFunction{T, CatTrace}
    is_indexed::Bool
    labels::Union{Nothing, Vector{T}}
    inverts_continuous::Bool
end
(LCat{T})(is_indexed::Bool, labels::Union{Nothing, Vector{T}}) where {T} = LCat{T}(is_indexed, labels, false)

labels(l::LCat, pvec) = l.is_indexed ? (1:length(pvec)) : l.labels
idx_to_label(c::LCat, idx) = c.is_indexed ? idx : c.labels[idx]
label_to_idx(c::LCat, lab) = c.is_indexed ? lab : findfirst([l == lab for l in c.labels])
Base.:(==)(a::LCat{T}, b::LCat{T}) where {T} =
    a.inverts_continuous == b.inverts_continuous && (a.is_indexed && b.is_indexed || a.labels == b.labels)

Cat = LCat{Int}(true, Int[])
ContinuousInvertingCat = LCat{Int}(true, Int[], true)
LCat(labels::Vector{T}) where {T} = LCat{T}(false, labels)
LCat(labels) = LCat(collect(labels))

Gen.simulate(c::LCat, (probs,)::Tuple) =
    let ps = recip_truncate(probs, c.inverts_continuous)
        CatTrace(c, probs, categorical(ps), nothing, nothing, ps)
    end
function Gen.generate(c::LCat, (probs,)::Tuple, cm::Union{Gen.ChoiceMap, Gen.EmptyChoiceMap})
    if isempty(cm)
        ps = fwd_truncate(probs, c.inverts_continuous)
        tr = CatTrace(c, probs, categorical(ps), nothing, nothing, nothing)
        return (
            tr,
            log(fwd_prob_estimate(tr)) + log(recip_prob_estimate(tr))
        )
    else
        @assert has_value(cm, :val)
        # @assert length(collect(get_submaps_shallow(cm))) == 0
        # @assert length(collect(get_values_shallow(cm))) == 1
        rscore = has_value(cm, :recip_score) ? cm[:recip_score] : nothing
        fscore = has_value(cm, :fwd_score) ? cm[:fwd_score] : nothing
        pprobs = has_value(cm, :proposal_probs) ? cm[:proposal_probs] : nothing
        idx = label_to_idx(c, cm[:val])
        @assert !isnothing(idx) "couldn't find value $(cm[:val])"
        tr = CatTrace(c, probs, idx, fscore, rscore, pprobs)
        return (tr, log(fwd_prob_estimate(tr)))
    end
end

function Gen.update(tr::CatTrace, (probs,)::Tuple, _::Tuple, cm::Gen.ChoiceMap)
    if isempty(cm) && probs == get_args(tr)[1]
        return (tr, 0., NoChange(), EmptyChoiceMap())
    else
        @assert isempty(cm) || has_value(cm, :val)
        newidx = isempty(cm) ? tr.idx : label_to_idx(get_gen_fn(tr), cm[:val])
        rscore = has_value(cm, :recip_score) ? cm[:recip_score] : nothing
        fscore = has_value(cm, :fwd_score) ? cm[:fwd_score] : nothing
        pprobs = has_value(cm, :proposal_probs) ? cm[:proposal_probs] : nothing
        newtr = CatTrace(get_gen_fn(tr), probs, newidx, fscore, rscore, pprobs)
        score =
            if weight_type() == :perfect
                get_score(newtr) - get_score(tr)
            elseif weight_type() == :noisy
       #         get_score(newtr) - get_score(tr)                
                # Return an unbiased estimate of P(x')/P(x)
                log(fwd_prob_estimate(newtr)) + log(recip_prob_estimate(tr)) 
            else
                error("`update` not implemented for this weight-type.")
            end

        return (
            newtr, score,
            tr.idx == newidx ? NoChange() : UnknownChange(),
            isempty(cm) ? EmptyChoiceMap() : StaticChoiceMap((val=get_retval(tr),), (;))
        )
    end
end

### Can something strange happen here due to being in the opposite weight mode?
### For instance when we use `assess` to calculate the reciprocal weight during PG
function fwd_prob_estimate(tr::CatTrace)
    if get_gen_fn(tr).inverts_continuous
        @assert weight_type() in (:recip, :perfect) "Error: Attempt to get a forward P-estimate for a `Cat` used to invert a discrete->continuous distribution in the model."
        if weight_type() == :perfect
            return normalize(tr.probs)[tr.idx]
        else
            @assert weight_type() == :recip
            return 1/recip_prob_for_continuous_inversion(tr)
        end
    else
        est = try
            fwd_prob_estimate(fwd_truncate(tr.probs)[tr.idx])
        catch e
            println(get_gen_fn(tr))
            println("probs: $(tr.probs)")
            throw(e)
        end
        maybe_put_est_into_trace!(tr, est, :fwd)
        return est
    end
end
function recip_prob_estimate(tr::CatTrace)
    if get_gen_fn(tr).inverts_continuous
        est = recip_prob_for_continuous_inversion(tr)
    else
        est = recip_prob_estimate(recip_truncate(tr.probs)[tr.idx])
    end
    maybe_put_est_into_trace!(tr, 1/est, :recip)
    return est
end

# est should be an estimate of p -- not 1/p
function maybe_put_est_into_trace!(tr, est, default_mode)
    if weight_type() == :perfect
        return
    end

    # I forget why there is a `isnothing` check in one case but not the other...is this right?
    if weight_type() == :fwd || (weight_type() == :noisy && default_mode == :fwd)
        tr.fwd_score = est
    elseif weight_type() == :recip || (weight_type() == :noisy && default_mode == :recip)
        if isnothing(tr.recip_score)
            tr.recip_score = 1/est
        end
    else
        error("The above if statements should be exhaustive. But got: default_mode = $default_mode ; weight_type() = $(weight_type())")
    end
end

function recip_prob_for_continuous_inversion(tr::CatTrace)
    # Let `p(xᶜ ; xᵈ)` denote the model discrete -> continuous distribution.

    # The proposal probability for x̃ᵈ is `p(xᶜ ; x̃ᵈ) / ∑_{xᵈ}{p(xᶜ ; x̃ᵈ)}`.
    # We want the overall weight output from the circuit to be an unbiased estimate of
    # `∑_{xᵈ}{p(xᶜ ; x̃ᵈ)}`.

    # Thus, we want this to output `estimate_of[ ∑_{xᵈ}{p(xᶜ ; x̃ᵈ)} ] / exactly[ p(xᶜ ; x̃ᵈ) ]`

    if isapprox(sum(tr.probs), 1.0)
        @warn "Warning: probs for continuous inversion appear to be normalized.  To get accurate score estimates, the unnormalized distribution should be used."
    end

    p_xᶜ_given_x̃ᵈ = tr.probs[tr.idx]
    
    sum_of_tuning_curves = sum(tr.probs)
    num_spikes_from_assemblies = poisson(sum_of_tuning_curves * ContinuousToDiscreteScoreNumSpikes())
    estimate_of_sum = num_spikes_from_assemblies / ContinuousToDiscreteScoreNumSpikes()

    est = estimate_of_sum / p_xᶜ_given_x̃ᵈ

    # println("p_xᶜ_given_x̃ᵈ = $p_xᶜ_given_x̃ᵈ ; sum_of_tuning_curves = $sum_of_tuning_curves ; sum_est = $estimate_of_sum ; est = $est")

    return est
end


Gen.has_argument_grads(::LCat) = (true,)
Gen.accepts_output_grad(::LCat) = false
function Gen.choice_gradients(
    tr::CatTrace, selection::Selection=EmptySelection(), retgrad=nothing
)
    @assert isnothing(retgrad)
    @assert isempty(selection)

    grad = zeros(length(tr.probs))
    grad[tr.idx] = 1/tr.probs[tr.idx]
    return ((grad,), EmptyChoiceMap(), EmptyChoiceMap())
end
Gen.accumulate_param_gradients!(trace::CatTrace, retgrad=nothing) =
    Gen.choice_gradients(trace, EmptySelection(), retgrad)[1]

function Gen.accumulate_param_gradients!(trace::CatTrace, retgrad=nothing, scale_factor=1.)
    @assert scale_factor == 1.
    return Gen.choice_gradients(trace, EmptySelection(), retgrad)[1]
end
