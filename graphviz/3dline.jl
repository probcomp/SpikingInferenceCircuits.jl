using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits

# TODO: CHANGEME: import real model and proposal
include("../experiments/3DLine/model.jl")
include("../experiments/3DLine/model_hyperparams.jl")

# TODO: CHANGEME: fill in the latent variables and obs variables and the domains
latent_domains() = (Vels(), Vels(), Vels(), Xs(), Ys(), Zs(), Rs(), ϕs(), θs())
obs_domains() = (θs(), ϕs())


# automatically compute some things:
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

# Load hyperparameter assignments, etc., for the spiking neural network compiler.
include("../experiments/utils/default_implementation_rules.jl")
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
    [:vxₜ, :vyₜ, :vzₜ, :xₜ, :yₜ, :zₜ, :rₜ, :exact_ϕ, :exact_θ, :obs_θ, :obs_ϕ],
    # order in which to feed latent variables into the step proposal
    [:obs_θ, :obs_ϕ],       # order in which to feed observations into the proposals
    
    # Order in which to recur variables:
    [:vxₜ, :vyₜ, :vzₜ, :xₜ, :yₜ, :zₜ, :rₜ, :exact_ϕ, :exact_θ],
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
