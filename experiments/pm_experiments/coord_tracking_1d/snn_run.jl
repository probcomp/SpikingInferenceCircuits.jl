using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

includet("model_proposal.jl")
include("pm_obs_models.jl")

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
include("../../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

### Log failure probability bound:
failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), 3, NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

latent_domains() = (Positions(), Vels())
obs_domains() = (Positions(),)
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 2

# Construct an SMC circuit, by telling each model the domains of the input variables
smc = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model_exact_pseudomarginalization, latent_domains()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
    GenFnWithInputDomains(step_proposal, latent_obs_domains()),
    [:xₜ, :vxₜ], # order in which to feed latent variables into the step proposal
    [:obsx],       # order in which to feed observations into the proposals
    [:xₜ, :vxₜ], # order in which to feed latent variables back into the step model for the next timestep
    NPARTICLES()
)
println("SMC Circuit Constructed.")

# Implement the circuit to a network of neurons.
impl = Circuits.memoized_implement_deep(smc, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs(
    RUNTIME(),
    INTER_OBS_INTERVAL(),
    [(obsx = obs,) for obs in [2, 4, 6, 17, 10, 11, 15, 16, 18, 19, 20]]
)
println("Constructed input spike sequence.")

# run the simulation.  returns a list of all spike events which occurred.  automatically serializes the events to disk
# after the simulation, unless the `save_events` kwarg is set to false.
# Give it the directory of this experiment so it saves in `experiments/this_experiment_directory/saves`.
# (If no dir is given, it will save in `experiments/saves`.)
events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__)
println("Simulation completed!")

# get the inferred latent states from the simulation
includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), 2 #= num latent vars in model =#)