module CPTs

using Distributions: Categorical, ncategories, probs
using Gen
import Distributions
import Gen

include("cpt.jl")
include("labeled_cpt.jl")

export CPT, LabeledCPT

end # module
