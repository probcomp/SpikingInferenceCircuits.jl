### Loss functions
normalize(vec) = vec/sum(vec)
KL(pv1, pv2) = sum(
    p1 == 0. ? 0. : p1 * (log(p1) - log(p2)) for (p1, p2) in zip(pv1, pv2)
)
onehot(x, size) = [x == y ? 1 : 0 for y=1:size]

### CPT --> Training data
assmt_to_onehots(assmt, dims) =
    vcat((onehot(v, dimsize) for (v, dimsize) in zip(assmt, dims))...)

function cpt_to_training_pairs(cpt)
    assmts = [assmt_to_onehots(Tuple(assmt), size(cpt)) for assmt in keys(cpt)] |> Flux.flatten |> x->reduce(hcat, x)
    vals   = collect(values(cpt)) |> Flux.flatten |> x->reduce(hcat, x)
    return (assmts, vals)
end
cpt_to_dataloader(cpt; batchsize=5) =
    Flux.DataLoader(cpt_to_training_pairs(cpt); batchsize)

### Model + loss function
insize(cpt) = sum(size(cpt))
outsize(cpt) = length(first(cpt))
simple_cpt_ann(cpt) = Chain(
    Dense(insize(cpt), 16, σ),
    Dense(16, 16, σ),
    Dense(16, outsize(cpt), σ)
)
simpler_cpt_ann(cpt) = Chain(Dense(insize(cpt), 8, σ), Dense(8, outsize(cpt), σ))
kl_loss(model) =
    (x, y) -> let
        modelprobs = model(x)
        p = normalize(modelprobs)

        # try to push to 0; penalize more for being too low than too high
        probsum = sum(modelprobs)
        normalization_penalty = probsum < 1 ? 5*(1 - probsum) : probsum - 1

        # for the fwd KL, set 0 probs to 1e-10 so we don't have things go to infinity
        kl_penalty = KL(p, map(x -> max(1e-10, x), y)) + KL(y, modelprobs)
        
        kl_penalty + normalization_penalty
    end