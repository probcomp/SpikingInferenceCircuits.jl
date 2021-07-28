using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

include("model.jl")
include("prior_proposal.jl")
@load_generated_functions()

latent_domains()     = (
    occₜ = positions(OccluderLength()),
    xₜ   = positions(SquareSideLength()),
    yₜ   = positions(SquareSideLength()),
    vxₜ  = Vels(),
    vyₜ  = Vels()
)
obs_domains() = (
    img_inner = SIC.DiscreteIRTransforms.ProductDomain([
        SIC.DiscreteIRTransforms.ProductDomain([
            SIC.DiscreteIRTransforms.EnumeratedDomain([true, false])
            for _=1:ImageSideLength()   
        ])
        for _=1:ImageSideLength()   
    ]),
)

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

# for (i, (gf, op)) in enumerate((
#     (GenFnWithInputDomains(step_latent_model, latent_domains()), Assess()),
#     (GenFnWithInputDomains(obs_model, latent_domains()), Assess()),
#     (GenFnWithInputDomains(_initial_prior_proposal, obs_domains()), Propose()),
#     (GenFnWithInputDomains(_step_prior_proposal, latent_obs_domains()), Propose())
# ))
#     println("getting gfc for $i")
#     SIC.gen_fn_circuit(gf, op)
#     println("finished $i")
# end

smccircuit = SMC(
    GenFnWithInputDomains(init_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(_initial_prior_proposal, obs_domains()),
    GenFnWithInputDomains(_step_prior_proposal, latent_obs_domains()),
    [:occₜ, :xₜ, :yₜ, :vxₜ, :vyₜ],
    [
        # To feed the image observation to the proposal, we need to tell it how to translate
        # from the observation trace addresses to the observed value
        # [In future versions, this should be done via the return value from the obs model]
        SIC.WithExtensionMap(
            :img_inner,
            trace_addr -> begin
                # trace addr will be
                # x => y => :got_photon
                (x, (y, rest)) = trace_addr
                @assert rest == :got_photon
                x => y # input to proposal using `x => y` [strip the `:got_photon`]
            end
        )
    ],
    [:occₜ, :xₜ, :yₜ, :vxₜ, :vyₜ],
    NPARTICLES();
    truncation_minprob=MinProb()
)
println("SMC Circuit Constructed.")

impl = Circuits.memoized_implement_deep(smccircuit, Spiking());
println("Circuit fully implemented using Poisson Process neurons.")

includet("../utils/simulation_utils.jl")
# # TODO: set up run