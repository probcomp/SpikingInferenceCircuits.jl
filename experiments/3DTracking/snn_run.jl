using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

include("model.jl")
include("model_hyperparams.jl")

latent_domains() = (Vels(), Vels(), Vels(), Xs(), Ys(), Zs(), Rs(), ϕs(), θs())
obs_domains() = (θs(), ϕs())


# automatically compute some things:
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
include("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
# Things you set:
NEURAL_NSTEPS() = 2
NEURAL_NPARTICLES() = 2
# don't change this:
RUNTIME() = INTER_OBS_INTERVAL() * (NEURAL_NSTEPS() - 0.1)


function extract_angle_indices(gt_tr)
    gt_obs_choices = get_choices(gt_tr)
    θ1 = findfirst(map(x -> x == gt_obs_choices[:init => :obs => :obs_θ => :val], θs()))
    ϕ1 = findfirst(map(x -> x == gt_obs_choices[:init => :obs => :obs_ϕ => :val], ϕs()))
    obs_list = [(θ1, ϕ1)]                   
    for step in 1:NEURAL_NSTEPS()
        obs_θ = findfirst(map(x -> x == gt_obs_choices[:steps => step => :obs => :obs_θ => :val], θs()))
        obs_ϕ = findfirst(map(x -> x == gt_obs_choices[:steps => step => :obs => :obs_ϕ => :val], ϕs()))
        push!(obs_list, (obs_θ, obs_ϕ))
    end
    return obs_list
end
                       
### Log failure probability bound:
failure_prob_bound = bound_on_overall_failure_prob(NEURAL_NSTEPS(), NVARS(), NEURAL_NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

# Construct an SMC circuit, by telling each model the domains of the input variables
smc = SMC(
    # TODO: CHANGEME: put in real model names
    GenFnWithInputDomains(initial_model, ()),
    GenFnWithInputDomains(step_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(initial_proposal, obs_domains()),
    GenFnWithInputDomains(step_proposal, latent_obs_domains()),

    # TODO: CHANGEME: replace with your latent and obs names
    # Order in which to feed in variables to proposal
    [:dx, :dy, :dz, :x, :y, :z, :r, :true_ϕ, :true_θ, :obs_ϕ, :obs_θ],
    # order in which to feed latent variables into the step proposal
    [:obs_ϕ, :obs_θ],       # order in which to feed observations into the proposals
    
    # Order in which to recur variables:
    [:dx, :dy, :dz, :x, :y, :z, :r, :true_ϕ, :true_θ],
    # order in which to feed latent variables back into the step model for the next timestep
    
    NEURAL_NPARTICLES();
    
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

include("../utils/simulation_utils.jl")

# `inputs` will be a vector specifying where to send inputs into the SNN at what time
inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms

    # TODO: CHANGEME: put in the actual observations you want to run on
    # Make sure they are indexed via 1...N, not some other domain (like, e.g. radians)

    [          # vector giving the observations at each timestep.
               # at each timestep, give a named tuple mapping observation names to observation values
               # Enough observation values must be specified to send one in at each timestep until the end of the simulation
               # (and more may be provided "to be save")
               # If an observed value comes from a domain other than {1, ..., N},
               # the observations must be fed in as the indexed version (ie. for the first value of the domain, feed in "1";
               # for the second value, "2", and so on). TODO: support giving observations in their true domains.
        (obs_θ = θ, obs_ϕ = ϕ)
               for (θ, ϕ) in extract_angle_indices(tr)




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
include("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NEURAL_NPARTICLES(), NLATENTS() #= num latent vars in model =#)



# domains = SIC.DiscreteIRTransforms.get_domains(
#     SIC.DiscreteIRTransforms.get_ir(SIC.DiscreteIRTransforms.replace_return_node(step_model)).nodes, latent_domains())


