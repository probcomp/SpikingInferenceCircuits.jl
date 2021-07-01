using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using Test

includet("model_proposal.jl")

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
include("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

### Log failure probability bound:
failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), 6, NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

### Construct the circuit:
latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)

# Construct an SMC circuit, by telling each model the domains of the input variables

rsmcstep = RecurrentSMCStep(
    SMCStep(
        GenFnWithInputDomains(step_latent_model, latent_domains()),
        GenFnWithInputDomains(obs_model, latent_domains()),
        GenFnWithInputDomains(step_proposal, latent_obs_domains()),
        [:xₜ, :vxₜ, :yₜ, :vyₜ],
        [:obsx, :obsy],
        NPARTICLES()
    ),
    [:xₜ, :vxₜ, :yₜ, :vyₜ]
)
println("SMC Circuit Constructed.")

# Implement the circuit to a network of neurons.
impl = Circuits.memoized_implement_deep(rsmcstep, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")

# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs_with_initial_latents(
    RUNTIME(), INTER_OBS_INTERVAL(),
    (
        xₜ₋₁ = 2, yₜ₋₁ = 7,
        vxₜ₋₁ = 1 + (1 - first(Vels())),
        vyₜ₋₁ = -1 + (1 - first(Vels()))
    ), [
        (obsx = x, obsy = y)
        for (x, y) in [
            (2, 8), (3, 7), (4, 5), (4, 4),
            (6, 4), (6, 3), (7, 2), (8, 1),
            (8, 1), (6, 1), (8, 1), (8, 2)
        ]
    ],
    NPARTICLES()
)

println("Constructed input spike sequence.")

# run the simulation.  returns a list of all spike events which occurred.  automatically serializes the events to disk
# after the simulation, unless the `save_events` kwarg is set to false.
# Give it the directory of this experiment so it saves in `experiments/this_experiment_directory/saves`.
# (If no dir is given, it will save in `experiments/saves`.)
events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__)
println("Simulation completed!")

includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES(), 4 #= num latent vars in model =#)
@test length(inferred_states) == floor(RUNTIME()/INTER_OBS_INTERVAL())