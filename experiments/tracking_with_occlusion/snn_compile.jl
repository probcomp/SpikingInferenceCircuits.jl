using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

include("model.jl")
include("prior_proposal.jl")
include("groundtruth_rendering.jl")
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
NOBS()     = length(obs_domains()) # TODO: this is wrong
NVARS()    = NLATENTS() + NOBS()

includet("../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")
println("BUT! Note that I have not tuned this check correctly to account for the image observations!")

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

matrix_to_vec_of_vecs(matrix) = [reshape(matrix[:, x], (:,)) for x=1:size(matrix)[2]] # matrix is indexed [y, x]
occ = [2, 2, 2]; x = [1, 1, 1]; y = [3, 2, 1]; vx = [0, 0, 0]; vy = [-1, -1, -1]
imgs = [matrix_to_vec_of_vecs(image_determ(args...)) for args in zip(occ, x, y)]

inputs = get_smc_circuit_inputs(
    RUNTIME(), # number of ms to simulate for
    INTER_OBS_INTERVAL(),      # send in a new observation every 1000 ms
    [
        [
            :img_inner => x => y => :got_photon => (img[x][y] == 0 ? 2 : 1)
            for x=1:length(img)
                for y=1:length(img[x])
        ]
        for img in imgs
    ]
)
println("Constructed input spike sequence.")

events = simulate_and_get_events(impl, RUNTIME(), inputs; dir=@__DIR__);
println("Simulation completed!")

includet("../utils/spiketrain_utils.jl")
inferred_states = get_smc_states(events, NPARTICLES())