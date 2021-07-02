using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

includet("model_proposal.jl")

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
includet("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

### Log failure probability bound:
failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), 6, NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)

# Construct an SMC circuit, by telling each model the domains of the input variables
smc = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
    GenFnWithInputDomains(step_proposal, latent_obs_domains()),
    [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables into the step proposal
    [:obsx, :obsy],       # order in which to feed observations into the proposals
    [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables back into the step model for the next timestep
    NPARTICLES()
)
println("SMC Circuit Constructed.")

# Implement the circuit to a network of neurons.
impl = Circuits.memoized_implement_deep(smc, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms
    [          # vector giving the observations at each timestep.
               # at each timestep, give a named tuple mapping observation names to observation values
               # Enough observation values must be specified to send one in at each timestep until the end of the simulation
               # (and more may be provided "to be save")
               # If an observed value comes from a domain other than {1, ..., N},
               # the observations must be fed in as the indexed version (ie. for the first value of the domain, feed in "1";
               # for the second value, "2", and so on). TODO: support giving observations in their true domains.
        (obsx = x, obsy = y)
        for (x, y) in [
            (2, 8), (3, 7), (4, 5), (4, 4),
            (6, 4), (6, 3), (7, 2), (8, 1),
            (8, 1), (6, 1), (8, 1), (8, 2)
        ]
    ]
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
inferred_states = get_smc_states(events, NPARTICLES(), 4 #= num latent vars in model =#)