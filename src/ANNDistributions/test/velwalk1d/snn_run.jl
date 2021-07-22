using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

include("../../../../experiments/velwalk1d/model.jl")
include("../../../../experiments/velwalk1d/inference.jl")
include("../../../../experiments/velwalk1d/visualize.jl")

latent_domains()     = (xₜ=Positions(), vₜ=Vels())
obs_domains()         = (obs=Positions(),)
latent_obs_domains() = (latent_domains()..., obs_domains()...)

NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

includet("../../../../experiments/utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")
println("Warning: the above hyperparameter checks do not account for the use of an artificial neural network in the proposal!")

smccircuit = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(_exact_init_proposal, obs_domains()),
    GenFnWithInputDomains(_ann_step_proposal, latent_obs_domains()),
    [:xₜ, :vₜ], [:obs], [:xₜ, :vₜ], NPARTICLES();
    truncation_minprob=MinProb(),
    rejuv_proposal=GenFnWithInputDomains(mh_kernel, latent_obs_domains())
)
println("SMC Circuit Constructed.")

impl = Circuits.memoized_implement_deep(smccircuit, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../../../../experiments/utils/simulation_utils.jl")

inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms
    [
        (obs = x,)
        for x in [16, 15, 11, 10, 8, 14, 11, 17, 18, 18, 18]
    ]
    # the above is an observation sequence sampled for ground truth
    # xₜ = [16, 14, 12, 10, 8, 10, 12, 14, 16, 19, 20]
    # vₜ = [-2, -2, -2, -2, -2, 2, 2, 2, 2, 3, 1]
)
println("Constructed input spike sequence.")

events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__)
println("Simulation completed!")

includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), NLATENTS())