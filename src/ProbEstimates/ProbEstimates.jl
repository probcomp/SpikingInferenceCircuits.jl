module ProbEstimates
using Gen
using Distributions

struct CatTrace <: Gen.Trace
    gf::Gen.GenerativeFunction
    probs::Vector{Float64}
    idx::Int
end
Gen.get_args(tr::CatTrace) = (tr.probs,)
Gen.get_retval(tr::CatTrace) = idx_to_label(tr.gf, tr.idx)
Gen.get_choices(tr::CatTrace) = StaticChoiceMap((val=get_retval(tr),), (;))
Gen.get_score(tr::CatTrace) = -log(recip_prob_estimate(tr))
Gen.get_gen_fn(tr::CatTrace) = tr.gf

# Not totally sure this is right--
Gen.project(tr::CatTrace, ::EmptySelection) = 0.

struct LCat{T} <: Gen.GenerativeFunction{T, CatTrace}
    is_indexed::Bool
    labels::Union{Nothing, Vector{T}}
end
idx_to_label(c::LCat, idx) = c.is_indexed ? idx : c.labels[idx]
label_to_idx(c::LCat, lab) = c.is_indexed ? lab : findfirst(c.labels .== lab)
Base.:(==)(a::LCat{T}, b::LCat{T}) where {T} = (
        a.is_indexed && b.is_indexed || a.labels == b.labels
    )

Cat = LCat{Int}(true, Int[])
LCat(labels::Vector{T}) where {T} = LCat{T}(false, labels)
LCat(labels) = LCat(collect(labels))

Gen.simulate(c::LCat, (probs,)::Tuple) = CatTrace(c, probs, categorical(probs))
function Gen.generate(c::LCat, (probs,)::Tuple, cm::Union{Gen.ChoiceMap, Gen.EmptyChoiceMap})
    if isempty(cm)
        tr = simulate(c, (probs,))
        return (
            tr,
            log(fwd_prob_estimate(tr)) + log(recip_prob_estimate(tr))
        )
    else
        @assert has_value(cm, :val)
        @assert length(collect(get_submaps_shallow(cm))) == 0
        @assert length(collect(get_values_shallow(cm))) == 1
        tr = CatTrace(c, probs, label_to_idx(c, cm[:val]))
        return (tr, log(fwd_prob_estimate(tr)))
    end
end

function Gen.update(tr::CatTrace, (probs,)::Tuple, _::Tuple, cm::Gen.ChoiceMap)
    if isempty(cm)
        @assert probs == get_args(tr)[1]
        return (tr, 0., NoChange(), EmptyChoiceMap())
    else
        error("Not expecting nontrivial `update` to be called on `Cat`.  Probs: $probs, cm: $cm")
    end
end

include("prob_est_hyperparameters.jl")

K_fwd() = Latency() * AssemblySize() * MaxRate() |> Int ∘ round
K_recip() = Latency() * (MaxRate() * MinProb()) * AssemblySize() |> Int ∘ round

fwd_prob_estimate(tr::CatTrace) = fwd_prob_estimate(tr.probs[tr.idx])
recip_prob_estimate(tr::CatTrace) = recip_prob_estimate(tr.probs[tr.idx])

acceptable_p_error(n_vars_in_model, err_constant=10) = K_fwd() > log(err_constant * n_vars_in_model) / MinProb()

# defines `fwd_prob_estimate` and `recip_prob_estimate`
# and utils for changing how they work to use perfect vs noisy weights.
include("weight_mode_switching.jl")

export LCat, Cat

end # module