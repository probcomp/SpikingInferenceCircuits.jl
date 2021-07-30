using Gen, ProbEstimates, DynamicModels
ProbEstimates.use_perfect_weights!() # initially we will test this in vanilla Gen
include("model.jl")

tree(tr) = tr[:init => :latents][1]
vals(tr) = [tr[:init => :obs][1], (tr[:steps => t => :obs][1] for t=1:get_args(tr)[1])...]

tr, weight = generate(model, (5,))
(tree(tr), vals(tr))