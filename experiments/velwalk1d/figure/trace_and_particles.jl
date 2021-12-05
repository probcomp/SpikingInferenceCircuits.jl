ProbEstimates.use_noisy_weights!()
tr = generate(model_contobs, (2,), choicemap(
    (:init => :latents => :xₜ => :val, 5),
    (:init => :latents => :vₜ => :val, 0),
    (:steps => 1 => :latents => :xₜ => :val, 5),
    (:steps => 1 => :latents => :vₜ => :val, 0),
    (:steps => 2 => :latents => :xₜ => :val, 5),
    (:steps => 2 => :latents => :vₜ => :val, 0),

    (:init => :obs => :yᶜₜ => :val, 4.1),
    (:steps => 1 => :obs => :yᶜₜ => :val, 5.1),
    (:steps => 2 => :obs => :yᶜₜ => :val, 4.9),
))[1]
yᶜs = [obs_choicemap(tr, t)[:yᶜₜ => :val] for t=0:2]

sample_disc(yᶜₜ) = categorical(exp.([tuning_curve(pos, yᶜₜ) for pos in Positions()]) |> normalize)
inferred_trs = predetermined_smc_contobs(tr, 3, exact_init_proposal_contobs, approx_step_proposal_contobs,
    (
        [
            choicemap((:init => :latents => :yᵈₜ => :val, sample_disc(yᶜs[1])), (:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, 0)),
            choicemap((:init => :latents => :yᵈₜ => :val, sample_disc(yᶜs[1])), (:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, -1)),
            choicemap((:init => :latents => :yᵈₜ => :val, sample_disc(yᶜs[1])), (:init => :latents => :xₜ => :val, 5), (:init => :latents => :vₜ => :val, 1))
        ],
        [
            [
                choicemap((:steps => 1 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[2])), (:steps => 1 => :latents => :xₜ => :val, 5), (:steps => 1 => :latents => :vₜ => :val, 0)),
                choicemap((:steps => 1 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[2])), (:steps => 1 => :latents => :xₜ => :val, 4), (:steps => 1 => :latents => :vₜ => :val, -1)),
                choicemap((:steps => 1 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[2])), (:steps => 1 => :latents => :xₜ => :val, 6), (:steps => 1 => :latents => :vₜ => :val, 1))
            ],
            [
                choicemap((:steps => 2 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[3])), (:steps => 2 => :latents => :xₜ => :val, 5), (:steps => 2 => :latents => :vₜ => :val, 0)),
                choicemap((:steps => 2 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[3])), (:steps => 2 => :latents => :xₜ => :val, 3), (:steps => 2 => :latents => :vₜ => :val, -1)),
                choicemap((:steps => 2 => :latents => :yᵈₜ => :val, sample_disc(yᶜs[3])), (:steps => 2 => :latents => :xₜ => :val, 6), (:steps => 2 => :latents => :vₜ => :val, 0))
            ]
        ]
    );
    ess_threshold=-Inf # no resampling
)[2];
nspikes = [1,1,1];