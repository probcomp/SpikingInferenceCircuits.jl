using Revise
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
using DynamicModels

includet("../experiments/velwalk1d/model.jl")
includet("../experiments/velwalk1d/pm_model.jl")
includet("../experiments/velwalk1d/inference.jl")
@load_generated_functions()


latent_domains()     = (xₜ=Positions(), vₜ=Vels())
obs_domains()         = (obs=Positions(),)

latent_obs_domains() = (latent_domains()..., obs_domains()...)
NLATENTS() = length(latent_domains())
NOBS()     = length(obs_domains())
NVARS()    = NLATENTS() + NOBS()

includet("../experiments/utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

### Run-specific hyperparams:
NSTEPS() = 2
RUNTIME() = INTER_OBS_INTERVAL() * (NSTEPS() - 0.1)
NPARTICLES() = 2

failure_prob_bound = bound_on_overall_failure_prob(NSTEPS(), NVARS(), NPARTICLES())
println("Hyperparameters set so the probability the circuit fails due to an issue we check for is less than $failure_prob_bound.")

smccircuit = SMC(
    GenFnWithInputDomains(initial_latent_model, ()),
    GenFnWithInputDomains(step_latent_model, latent_domains()),
    GenFnWithInputDomains(obs_model, latent_domains()),
    GenFnWithInputDomains(_exact_init_proposal, obs_domains()),
    GenFnWithInputDomains(_approx_step_proposal, latent_obs_domains()),
    [:xₜ, :vₜ], [:obs], [:xₜ, :vₜ], NPARTICLES();
    truncation_minprob=MinProb()
)
println("SMC Circuit Constructed.")

impl1 = implement(smccircuit, Spiking())
step1 = implement(impl1[:initial_step], Spiking())
is_particle = implement(step1.subcomponents[:particles], Spiking()).subcomponents[1];
is_impl = implement(is_particle, Spiking());

# propose_impl = Circuits.memoized_implement_deep(is_impl[:propose], Spiking())
# impl = Circuits.memoized_implement_deep(smccircuit, Spiking());

### neuron viz:
# inlined, names, state = Circuits.inline(impl)

# Circuits.viz!(inlined; base_path="graphviz/velwalk1d")

### Stuff for SDC viz:
using SpikingInferenceCircuits.SDCs: Mux, 
                                     ConditionalScore,
                                     ConditionalSample, 
                                     ToAssmts,
                                     LookupTable,
                                     NonnegativeRealMultiplier,
                                     Theta,
                                     Step,
                                     MultiInputLookupTable,
                                     ValueBlockerPasser

sdcs = [Mux, 
        ConditionalScore, 
        ConditionalSample, 
        ToAssmts, 
        LookupTable, 
        NonnegativeRealMultiplier, 
        Theta, 
        Step, 
        #CPTSampleScore, 
        MultiInputLookupTable, 
        ValueBlockerPasser
]
function inline_node_check(component)
    try
        return any(Circuits.has_abstract_of_type(component, t) for t in sdcs)
    catch e
        return false
    end
end

function Circuits.add_node!(g, name, cc::CompositeComponent)
    i = findfirst([has_abstract_of_type(cc, t) for t in sdcs])
    type = sdcs[i]
    node_attrs = Dict{Symbol, Any}(:shape => "square",
                                   :label => repr(type))
    Circuits.CircuitViz.add_vertex!(g, node_attrs)

end


### Now make the dot files!

is_impl_deep = Circuits.memoized_implement_deep(is_impl, Spiking());

### SDC: 
scd_inlined, sdc_names, sdc_state = Circuits.inline(is_impl_deep;
                                        treat_as_primitive = inline_node_check)
try
    Circuits.viz!(sdc_state; 
              base_path = "graphviz/sdc_velwalk1d_is")
catch e
    @warn "caught $e"
end

### Neuron
neuron_inlined, neuron_names, neuron_state = Circuits.inline(is_impl_deep)
try
    Circuits.viz!(neuron_state; 
              base_path = "graphviz/neuron_velwalk1d_is")
catch e
    @warn "caught $e"
end