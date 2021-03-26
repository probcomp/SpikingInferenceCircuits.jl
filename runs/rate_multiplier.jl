import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator
using Distributions: Categorical

includet("../src/value_types.jl")
includet("../src/components/real_multiplication/rate_multiplier.jl")

mult = RateMultiplier(4.0, 24.0, (SpikeRateReal(2.0), SpikeRateReal(3.0)))
poisson2 = OnOffPoissonNeuron(2.0)
poisson3 = OnOffPoissonNeuron(9.0)

circuit = CompositeComponent(
    CompositeValue((on=SpikeWire(),)), CompositeValue((out=SpikeWire(),)),
    (;mult, poisson2, poisson3),
    (
        CompOut(:poisson2, :out) => CompIn(:mult, 1),
        CompOut(:poisson3, :out) => CompIn(:mult, 2),
        Input(:on) => CompIn(:poisson2, :on),
        Input(:on) => CompIn(:poisson3, :on),
        CompOut(:mult, :out) => Output(:out)
    )
)
implemented = implement_deep(circuit, Spiking())

RUN_LEN() = 100.0
events = Sim.simulate_for_time_and_get_events(implemented,  RUN_LEN(); initial_inputs=(:on,))
println("Finished running simulation.")

function spiketrain_dict(event_vector)
    spiketrains = Dict()
    for (time, compname, outspike) in event_vector
        if haskey(spiketrains, compname)
            push!(spiketrains[compname], time)
        else
            spiketrains[compname] = [time]
        end
    end
    return spiketrains
end

using SpikingCircuits.SpiketrainViz

is_primitive_output(compname, event) = (
    event isa Sim.OutputSpike && compname in (:poisson2, :poisson3, :mult)
)
dict = spiketrain_dict(filter(((t,args...),) -> is_primitive_output(args...), events))
# draw_spiketrain_figure(
#     collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
# )

avg_mult_rate = length(dict[:mult]) / RUN_LEN()

state_changes = filter(events) do (_, compname, evt)
    evt isa Sim.StateChange && compname == :mult
end

function expected_rate(state_changes)
    prev_t = state_changes[1][1]
    prev_st = state_changes[1][3].new_state
    int = 0
    for (new_t, _, stch) in state_changes[2:end]
        new_st = stch.new_state
        ΔT = new_t - prev_t
        int += ΔT * SpikingCircuits.rate(implement(mult, Spiking())[:neuron], prev_st.substates[:neuron])

        prev_t, prev_st = (new_t, new_st)
    end
    return int / RUN_LEN()
end

(avg_mult_rate, expected_rate(state_changes))
