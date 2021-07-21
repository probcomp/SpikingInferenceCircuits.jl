module ANNDistributions
using Flux

include("train_on_cpt.jl")

export simple_cpt_ann, kl_loss, cpt_to_dataloader

end # module
