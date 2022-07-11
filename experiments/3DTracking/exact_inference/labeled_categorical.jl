struct LabeledCategorical{T} <: Gen.Distribution{T}
    labels::Vector{T}
    val_to_idx::Dict{T, Int}

    function LabeledCategorical(v::Vector{T}) where T
        val_to_idx = Dict{T, Int}()
        for (i, v) in enumerate(v)
            @assert !haskey(val_to_idx, v)
            val_to_idx[v] = i
        end
        return new{T}(v, val_to_idx)
    end    
end
LabeledCategorical(v) = LabeledCategorical(collect(v))
Gen.random(l::LabeledCategorical, probs) = l.labels[categorical(probs)]
Gen.logpdf(l::LabeledCategorical, val, probs) = log(probs[l.val_to_idx[val]])