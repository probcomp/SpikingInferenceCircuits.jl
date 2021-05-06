using CPTs
using Distributions: ncategories, probs
assmt_cond_prob_matrix(cpt::CPT) =
    let a = assmts(cpt)
        (collect ∘ transpose)(hcat([
            probs(cpt[a[i]])
            for i=1:length(a)
        ]...))
    end

include("cpt_sample.jl")
include("cpt_score.jl")