include("dynamic_model.jl")

@gen (static) function dumb_step_proposal(xₜ₋₁, vxₜ₋₁, yₜ₋₁, vyₜ₋₁, obsx, obsy)
    xₜ ~ categorical(truncated_discretized_gaussian(obsx, 2.0, Positions()))
    yₜ ~ categorical(truncated_discretized_gaussian(obsy, 2.0, Positions()))
    vxₜ ~ categorical(onehot(
        truncate_value(truncate_value(xₜ - xₜ₋₁, (vxₜ₋₁ - 1):(vxₜ₋₁ + 1)), Vels()),
        Vels()
    ))
    vyₜ ~ categorical(onehot(
        truncate_value(truncate_value(yₜ - yₜ₋₁, (vyₜ₋₁ - 1):(vyₜ₋₁ + 1)), Vels()),
        Vels()
    ))
end
NSTEPS = 10
dm = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4)
smart_sp = @compile_step_proposal(step_proposal, 4, 2)
dumb_sp = @compile_step_proposal(dumb_step_proposal, 4, 2)
@load_generated_functions()

tr = simulate(dm, (NSTEPS,))
obss = get_dynamic_model_obs(tr)

smc_inferences = dynamic_model_smc(
    dm, obss, cm -> (cm[:obsx], cm[:obsy]),
    initial_proposal, smart_sp, 4
)

dumb_inferences = dynamic_model_smc(
    dm, obss, cm -> (cm[:obsx], cm[:obsy]),
    initial_proposal, dumb_sp, 1 # 1 particle so no the weights have no effect
)

smc_unw = smc_inferences[1];
dumb_unw = dumb_inferences[1];

avg_dist_from_ground_truth(inferred_trace, gt) = sum(
        dist(inferred_trace, gt, t)
        for t=1:get_args(inferred_trace)[1]
    )/get_args(inferred_trace)[1]

function dist(tr1, tr2, t)
    x1 = tr1[:steps => t => :latents => :xₜ]
    y1 = tr1[:steps => t => :latents => :yₜ]
    x2 = tr2[:steps => t => :latents => :xₜ]
    y2 = tr2[:steps => t => :latents => :yₜ]
    return sqrt((x1 - x2)^2 + (y1 - y2)^2)
end

# As a measure of inference quality (other than trace probability),
# we can check the average cartesian distance between the inferred
# object position and the true object position.
# (This seems like an important metric for, e.g., how likely it is
# an animal will successfully grab it's prey.)
s_dist = avg_dist_from_ground_truth(
    smc_unw[10][1], tr
)
d_dist = avg_dist_from_ground_truth(
    dumb_unw[10][1], tr
)

(s_dist, d_dist)