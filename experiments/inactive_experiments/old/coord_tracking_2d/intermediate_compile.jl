"""
This is a test for compiling syntax that is _almost_ the same as is used in Vanilla Gen.
I'm using it as I add features to the compilation, to test that they are working.
Eventually once the exact same syntax is supported we won't need this additional file.
"""

using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using Gen, Distributions
using ProbEstimates: Cat, LCat

include("modeling_utils.jl")
include("model_hyperparams.jl")

@gen (static) function step_model(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁)
    vxₜ ~ LCat(Vels())(maybe_one_off(vxₜ₋₁, 0.3, Vels()))
    vyₜ ~ LCat(Vels())(maybe_one_off(vyₜ₋₁, 0.3, Vels()))

    exp_x = xₜ₋₁ + vxₜ
    exp_y = yₜ₋₁ + vyₜ
    xₜ ~ Cat(maybe_one_off(exp_x, 0.3, Positions()))
    yₜ ~ Cat(maybe_one_off(exp_y, 0.3, Positions()))
    
    return (xₜ, vxₜ, yₜ, vyₜ)
end
@gen (static) function obs_model(xₜ, vxₜ, yₜ, vyₜ)
    obsx ~ Cat(truncated_discretized_gaussian(xₜ, 4.0, Positions()))
    obsy ~ Cat(truncated_discretized_gaussian(yₜ, 4.0, Positions()))

    return (obsx, obsy)
end
@gen (static) function step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    projected_x = truncate_value(xₜ₋₁ + vxₜ₋₁, Positions())
    mean_x = 0.5 * obsx + 0.5 * projected_x

    xₜ ~ Cat(
        # it is possible to be up to 2 away from the projected_x
        truncate_dist_to_valrange(
            discretized_gaussian(mean_x, 2.5, Positions()),
            (projected_x - 2):(projected_x + 2),
            Positions()
        ) |> truncate
    )
    
    diff_x = xₜ - xₜ₋₁
    vxₜ ~ LCat(Vels())(vel_step_dist(vxₜ₋₁, diff_x))

    projected_y = truncate_value(yₜ₋₁ + vyₜ₋₁, Positions())
    mean_y = (obsy + projected_y)/2
    yₜ ~ Cat(
        truncate_dist_to_valrange(
            discretized_gaussian(mean_y, 2.5, Positions()),
            (projected_y - 2):(projected_y + 2),
            Positions()
        ) |> truncate
    )
    diff_y = yₜ - yₜ₋₁
    vyₜ ~ LCat(Vels())(vel_step_dist(vyₜ₋₁, diff_y))
end

err_if_not_probvec(pvec, errmsg) =
    if isprobvec(pvec)
        pvec
    else
        error(errmsg)
    end

vel_step_dist(vxₜ₋₁, diff_x) =
    let probs = maybe_one_off(vxₜ₋₁, 0.3, Vels()) .* maybe_one_off(diff_x, 0.5, Vels())
        isprobvec(probs) ? probs : maybe_one_off(vxₜ₋₁, 0.3, Vels())
    end |> normalize

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 2

rsmcstep = RecurrentSMCStep(
    SMCStep(
        GenFnWithInputDomains(step_model, latent_domains()),
        GenFnWithInputDomains(obs_model, latent_domains()),
        GenFnWithInputDomains(step_proposal, latent_obs_domains()),
        [:xₜ, :vxₜ, :yₜ, :vyₜ],
        [:obsx, :obsy],
        NPARTICLES()
    ),
    [:xₜ, :vxₜ, :yₜ, :vyₜ]
)
println("Circuit constructed.")