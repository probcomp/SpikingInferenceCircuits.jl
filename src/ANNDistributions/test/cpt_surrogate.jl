using ANNDistributions
using Distributions
using Flux

normalize(pvec) = pvec/sum(pvec)
discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
])

Positions() = 1:20
Vels()      = -2:2
function probs(xₜ₋₁, vxₜ₋₁, obs)
    p_vx  = discretized_gaussian(vxₜ₋₁, 1.0, Vels())
    p_obs = [discretized_gaussian(xₜ₋₁ + vx, 1.0, Positions())[obs] for vx in Vels()]
    prod = p_vx .* p_obs
    unnormalized = sum(prod) != 0 ? prod : p_vx
    return normalize(unnormalized)
end

cpt = [
        probs(xₜ₋₁, vxₜ₋₁, obs)
        for xₜ₋₁ in Positions(), vxₜ₋₁ in Vels(), obs in Positions()
    ]

function train_a_model(; model=simple_cpt_ann(cpt), n_iters=10)
    dataloader = cpt_to_dataloader(cpt)
    loss = kl_loss(model)
    average(xs) = sum(xs)/length(xs)
    avg_loss() = average([loss(d...) for d in cpt_to_dataloader(cpt, batchsize=1)])
    for i=1:n_iters
        println("Beginning epoch $i; avg loss = $(avg_loss())")
        Flux.train!(
            kl_loss(model), Flux.params(model), dataloader, Flux.Optimise.Descent()
        )
    end
    return model
end
model = train_a_model()
model = train_a_model(;model, n_iters=30)