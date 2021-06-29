using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

includet("model_proposal.jl")

includet("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 2

# Construct an SMC circuit, by telling each model the domains of the input variables
# smc = SMC(
#     GenFnWithInputDomains(initial_latent_model, ()),
#     GenFnWithInputDomains(step_latent_model, latent_domains()),
#     GenFnWithInputDomains(obs_model, latent_domains()),
#     GenFnWithInputDomains(initial_proposal, obs_domains()),
#     GenFnWithInputDomains(step_proposal, latent_obs_domains()),
#     [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables into the step proposal
#     [:obsx, :obsy],       # order in which to feed observations into the proposals
#     [:xₜ, :vxₜ, :yₜ, :vyₜ], # order in which to feed latent variables back into the step model for the next timestep
#     NPARTICLES()
# )

# println("SMC Circuit Constructed.")

# impl = Circuits.memoized_implement_deep(smc, Spiking());
# println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

RUNTIME() = 2.0
# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    1000,      # send in a new observation every 1000 ms
    [          # vector giving the observations at each timestep.
               # at each timestep, give a named tuple mapping observation names to observation values
               # Enough observation values must be specified to send one in at each timestep until the end of the simulation
               # (and more may be provided "to be save")
               # If an observed value comes from a domain other than {1, ..., N},
               # the observations must be fed in as the indexed version (ie. for the first value of the domain, feed in "1";
               # for the second value, "2", and so on). TODO: support giving observations in their true domains.
        (obsx = x, obsy = y)
        for (x, y) in [
            (2, 16), (6, 14), (9, 11),
            (11, 10), (11, 7), (12, 6),
            (18, 6), (18, 4), (19, 1),
            (19, 1), (19, 1), (19, 1)
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