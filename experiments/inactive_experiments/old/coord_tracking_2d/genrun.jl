include("dynamic_model.jl")

NSTEPS = 10
model = @DynamicModel(initial_latent_model, step_latent_model, obs_model, 4)
proposal_init = @compile_initial_proposal(initial_proposal, 2)
proposal_step = @compile_step_proposal(step_proposal, 4, 2)
@load_generated_functions()

tr = simulate(model, (NSTEPS,))
obss = get_dynamic_model_obs(tr)

smc_inferences = dynamic_model_smc(
    model, obss, cm -> (cm[:obsx => :val], cm[:obsy => :val]),
    proposal_init, proposal_step, 20
)
