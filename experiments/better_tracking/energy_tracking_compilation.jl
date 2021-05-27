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

NPARTICLES() = 2

m = @compile step_model(Xs(), Bools(), Energies(), Vels(), Bools())
p =  @compile step_proposal(Xs(), Vels(), Bools(), Bools(), Energies(), Xs())
@infer function smcprog(m, p)
    loop(observe(:obsₜ), NPARTICLES()) do 
        is(m, p)
    end
end
smc_circuit = smcprog(m, p)
println("Circuit constructed.")

smc_impl = Circuits.memoized_implement_deep(smc_circuit, Spiking())
println("Circuit implemented.")

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


function do_run_may25_5pm()
    ins = get_inputs(12500, 1000,
        10, # initial x
        2 + 4, # initial v (add in offset by 4)
        7, # initial energy
        [9, 12, 13, 14, 15, 15, 16, 15, 15, 14, 13, 11, 9, 9, 9, 9, 9, 9],
    NPARTICLES())
    println("Got inputs.")
    events = get_events(smc_impl, 12500, ins)
    println("Simulation completed.")
    try
        serialize("energy_tracking_may25_5pm_events.jls", events)
        println("serialized")
    catch e
        @error("Error when trying to serialize")
    end
    return events
end

function do_run_may25_9_33pm(impl)
    ins = get_inputs(12500, 1000,
        10, # initial x
        2 + 4, # initial v (add in offset by 4)
        7, # initial energy
        [9, 12, 13, 14, 15, 15, 16, 15, 15, 14, 13, 11, 9, 9, 9, 9, 9, 9],
    NPARTICLES())
    println("Got inputs.")
    events = get_events(impl, 12500, ins)
    println("Simulation completed.")
    try
        serialize("energy_tracking_may25_9-33pm_events.jls", events)
        println("serialized")
    catch e
        @error("Error when trying to serialize")
    end
    return events
end