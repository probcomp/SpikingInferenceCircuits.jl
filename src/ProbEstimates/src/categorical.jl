mutable struct CatTrace{T} <: Gen.Trace
    gf::Gen.GenerativeFunction{T}
    probs::Vector{Float64}
    idx::Int
    fwd_score::Union{Nothing, Float64}
    recip_score::Union{Nothing, Float64}
end
CatTrace(gf, probs, idx) = CatTrace(gf, probs, idx, nothing, nothing)

Gen.get_args(tr::CatTrace) = (tr.probs,)
Gen.get_retval(tr::CatTrace) = idx_to_label(tr.gf, tr.idx)
Gen.get_choices(tr::CatTrace) = StaticChoiceMap((val=get_retval(tr), recip_score=tr.recip_score, fwd_score=tr.fwd_score), (;))
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
end
labels(l::LCat, pvec) = l.is_indexed ? (1:length(pvec)) : l.labels
idx_to_label(c::LCat, idx) = c.is_indexed ? idx : c.labels[idx]
label_to_idx(c::LCat, lab) = c.is_indexed ? lab : findfirst([l == lab for l in c.labels])
Base.:(==)(a::LCat{T}, b::LCat{T}) where {T} = (
        a.is_indexed && b.is_indexed || a.labels == b.labels
    )

Cat = LCat{Int}(true, Int[])
LCat(labels::Vector{T}) where {T} = LCat{T}(false, labels)
LCat(labels) = LCat(collect(labels))

Gen.simulate(c::LCat, (probs,)::Tuple) = CatTrace(c, probs, categorical(recip_truncate(probs)))
function Gen.generate(c::LCat, (probs,)::Tuple, cm::Union{Gen.ChoiceMap, Gen.EmptyChoiceMap})
    if isempty(cm)
        tr = CatTrace(c, probs, categorical(fwd_truncate(probs)))
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
        tr = CatTrace(c, probs, label_to_idx(c, cm[:val]), fscore, rscore)
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
        newtr = CatTrace(get_gen_fn(tr), probs, newidx, fscore, rscore)
        score =
            if weight_type() == :perfect
                get_score(newtr) - get_score(tr)
            elseif weight_type() == :noisy
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
    est = fwd_prob_estimate(fwd_truncate(tr.probs)[tr.idx])
    # @assert isnothing(tr.fwd_score)
    tr.fwd_score = est
    return est
end
function recip_prob_estimate(tr::CatTrace)
    est = recip_prob_estimate(recip_truncate(tr.probs)[tr.idx])
    if isnothing(tr.recip_score)
        tr.recip_score = est
    end
    return est
end