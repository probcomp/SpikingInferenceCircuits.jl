using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

includet("model.jl")
includet("inference.jl")

latent_domains()     = (xₜ=Positions(), yₜ=Positions(), vxₜ=Vels(), vyₜ=Vels(), vₜ=Vels2D())
latent_args_domains() = (xₜ=Positions(), yₜ=Positions(), vxₜ=Vels(), vyₜ=Vels())
obs_domains()         = (obsx=Positions(), obsy=Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)

NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

includet("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

 smccircuit = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_args_domains()),
    GenFnWithInputDomains(obs_model, latent_args_domains()),
    GenFnWithInputDomains(_exact_init_proposal, obs_domains()),
    GenFnWithInputDomains(_approx_step_proposal, (latent_args_domains()..., obs_domains()...)),
    [:xₜ, :yₜ, :vxₜ, :vyₜ], [:obsx, :obsy], [:xₜ, :yₜ, :vxₜ, :vyₜ], NPARTICLES();
    truncation_minprob=MinProb()
)
println("SMC Circuit Constructed.")

impl = Circuits.memoized_implement_deep(smccircuit, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms
    [
        (obsx = x, obsy = y)
        for (x, y) in [(2, 1), (3, 3), (4, 6), (5, 7), (5, 10), (7, 10), (5, 8), (3, 6), (1, 3), (2, 3), (1, 2)]
    ]
    # the above is an observation sequence sampled for ground truth
    # (xₜ, yₜ) = [(2, 1), (3, 3), (4, 5), (5, 7), (6, 9), (7, 10), (5, 8), (3, 6), (1, 4), (1, 3), (1, 2)]
    # vₜ = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (-2, -2), (-2, -2), (-2, -2), (-1, -1), (-1, -1)]
)
println("Constructed input spike sequence.")

events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__)
println("Simulation completed!")

includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), NLATENTS())