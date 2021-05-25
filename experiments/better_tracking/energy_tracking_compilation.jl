using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using CPTs
using DiscreteIRTransforms
using Distributions: Normal, cdf

includet("energy_tracking_model_proposal.jl")
includet("implementation_rules.jl")
includet("inference_dsl.jl")
includet("input_construction.jl")

m = @compile step_model(Xs(), Bools(), Energies(), Vels(), Bools())
p =  @compile step_proposal(Xs(), Vels(), Bools(), Bools(), Energies(), Xs())
@infer function smcprog(m, p)
    loop(observe(:obsₜ), NPARTICLES()) do 
        is(m, p)
    end
end
smc_circuit = smcprog(m, p)

# run!
include("../logging_utils.jl")
get_events(impl, runtime, inputs; log_interval=400) =
    SpikingSimulator.simulate_for_time_and_get_events(
        impl, runtime;
        inputs,
        log=true,
        log_filter=get_log_filter(log_interval),
        log_str=time_log_str
    )


NPARTICLES() = 10
ins = get_inputs(2500, 1000, 10, 2 + 4, 7, [11, 15, 14, 14], NPARTICLES())
println("Got inputs.")
events = get_events(impl_deep, 2500, ins)