using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

# TODO: CHANGEME: import real model and proposal
includet("model_proposal.jl")

# TODO: CHANGEME: fill in the latent variables and obs variables and the domains
latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())

# automatically compute some things:
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
includet("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
# Things you set:
NSTEPS() = 2
NPARTICLES() = 2
# don't change this:
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)

### Log failure probability bound:
failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

# Construct an SMC circuit, by telling each model the domains of the input variables
smc = SMC(
    # TODO: CHANGEME: put in real model names
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
    GenFnWithInputDomains(step_proposal, latent_obs_domains()),

    # TODO: CHANGEME: replace with your latent and obs names
    # Order in which to feed in variables to proposal
    [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables into the step proposal
    [:obsx, :obsy],       # order in which to feed observations into the proposals
    
    # Order in which to recur variables:
    [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables back into the step model for the next timestep
    
    NPARTICLES();
    
    # If you add this, the proposal will automatically be truncated so it never proposes a value
    # with proposal probablity < MinProb.
    truncation_minprob=MinProb()
)
# If it takes more than a few minutes to get to this following println,
# it could be trying to compile a huge CPT -- so we should debug what's happening.
println("SMC Circuit Constructed.")

# Implement the circuit to a network of neurons.
impl = Circuits.memoized_implement_deep(smc, Spiking()); # This will take a while [probably < 15 mins]
println("Circuit fully implemented using Poisson Process neurons.")

### Now the circuit is implemented, and we are going to run a simulation.

includet("../utils/simulation_utils.jl")

# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms

    # TODO: CHANGEME: put in the actual observations you want to run on
    # Make sure they are indexed via 1...N, not some other domain (like, e.g. radians)

    # { -1, -.9, ..., .9, 1 }
    # { 1, 2,   ...,  20, 21 }
    # Feed in observations from {1, ..., 21}, not {-1, ..., 1}

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
events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__) # This will take a long time [a day or two]
println("Simulation completed!")

# get the inferred latent states from the simulation
includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), NLATENTS() #= num latent vars in model =#)