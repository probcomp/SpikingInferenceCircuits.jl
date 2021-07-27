using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

include("../tracking/implementation_rules.jl")

binary() = EnumeratedDomain([true, false])

num_neurons(::Circuits.PrimitiveComponent{Spiking}) = 1
num_neurons(c::Circuits.CompositeComponent) = sum(num_neurons(sc) for sc in c.subcomponents)

N_VARS() = 4
@gen (static) function model(
    prev_x1, prev_x2, prev_x3, prev_x4
)
    x1 ~ bernoulli(prev_x1 ? 0.7 : 0.3)
    x2 ~ bernoulli(x1 && prev_x2 ? 0.7 : 0.3)
    x3 ~ bernoulli(x2 && prev_x3 ? 0.7 : 0.3)
    x4 ~ bernoulli(x3 && prev_x4 ? 0.7 : 0.3)

    obs1 ~ bernoulli(x1 ? 0.8 : 0.2)
    obs2 ~ bernoulli(x2 ? 0.8 : 0.2)
    obs3 ~ bernoulli(x3 ? 0.8 : 0.2)
    obs4 ~ bernoulli(x4 ? 0.8 : 0.2)

    return obs1 # return value doesn't matter
end
model_with_cpts, _ = to_indexed_cpts(model, [binary() for _=1:N_VARS()])

@gen (static) function proposal(
    prev_x1, prev_x2, prev_x3, prev_x4,
    obs1, obs2, obs3, obs4
)
    x1 ~ bernoulli(prev_x1 ? 0.7 : 0.3)
    x2 ~ bernoulli(x1 && prev_x2 ? 0.7 : 0.3)
    x3 ~ bernoulli(x2 && prev_x3 ? 0.7 : 0.3)
    x4 ~ bernoulli(x3 && prev_x4 ? 0.7 : 0.3)

    return x1 # return value doesn't matter
end
proposal_with_cpts, _ = to_indexed_cpts(proposal, [binary() for _=1:2*N_VARS()])

@load_generated_functions()

function get_num_neurons(model_with_cpts, proposal_with_cpts, N_VARS, NPARTICLES=4)
    dom2 = FiniteDomain(2)
    model_in_domains = Tuple(dom2 for _=1:N_VARS)
    proposal_in_domains = Tuple(dom2 for _=1:(2*N_VARS))
    
    
    smc_circuit = SIC.SMC(NPARTICLES, model_with_cpts, proposal_with_cpts, model_in_domains, proposal_in_domains)
    println("SMC Circuit Constructed.")
    
    smc_impl = Circuits.memoized_implement_deep(smc_circuit, Spiking())
    println("SMC Circuit Implemented.")
    
    return num_neurons(smc_impl)
end

cnt = get_num_neurons(model_with_cpts, proposal_with_cpts, N_VARS())