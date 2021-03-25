import JSON
using Gen
using Circuits
using SpikingCircuits

includet("../components/value_types.jl")
includet("../components/mux/mux.jl")
includet("../components/ipoisson_gated_repeater.jl")
includet("../components/mux/int_poisson_mux.jl")
includet("../components/cvb.jl")
includet("../components/conditional_sample_score/abstract.jl")
includet("../components/conditional_sample_score/spiking.jl")
includet("../components/thresholded_spike_counter.jl")
includet("../components/to_assmts/abstract.jl")
includet("../components/to_assmts/spiking.jl")
includet("../components/cpt.jl")
includet("../components/cpt_sample_score/abstract.jl")
includet("../components/cpt_sample_score/spiking.jl")
includet("../components/real_multiplication/abstract.jl")
includet("../components/real_multiplication/rate_multiplier.jl")
includet("../compiler/compiler.jl")

@gen (static) function test(in)
    x ~ CPT([[0.5, 0.5]])(in)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    return y
end

circuit = propose_circuit(test, (1,))

### implement the circuit ###

REF_RATE() = 1.0
OFF_RATE() = 0.0001
ON_RATE() = 2.0

Circuits.implement(p::PositiveReal, ::Spiking) =
    SpikeRateReal(REF_RATE())
Circuits.implement(p::PositiveRealMultiplier, ::Spiking) =
    RateMultiplier(
        6.0, REF_RATE(),
        Tuple(SpikeRateReal(REF_RATE()) for _=1:p.n_inputs)
    )
Circuits.implement(c::CPTSampleScore, ::Spiking) =
    SpikingCPTSampleScore(c, OFF_RATE(), ON_RATE())

implemented1 = implement(implement(circuit, Spiking()), Spiking())
implemented2 = implement_deep(implemented1, Spiking())

println("Component implemented.")

### visualize ###

includet("../visualization/circuit_visualization/component_interface.jl")

open("visualization/circuit_visualization/frontend/renders/gen_fn.json", "w") do f
    JSON.print(f, viz_graph(implemented1), 2)
end
println("Wrote component viz file.")

# ### simulate ###

events = SpikingSimulator.simulate_for_time_and_get_events(implemented2, 20.0; initial_inputs=(1 => 1,))

println("Simulation complete.")

# ### spiketrain ###
function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, _, outspike) in event_vector
        if haskey(spiketrains, outspike.name)
            push!(spiketrains[outspike.name], time)
        else
            spiketrains[outspike.name] = [time]
        end
    end
    return spiketrains
end

includet("../visualization/spiketrain.jl")
using .SpiketrainViz

is_primary_output(compname, event) = (isnothing(compname) && event isa SpikingSimulator.OutputSpike)
dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))
draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)