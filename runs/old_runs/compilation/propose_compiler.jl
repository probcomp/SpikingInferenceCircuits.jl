import JSON
using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

@gen (static) function test(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    z ~ CPT([
        [[0.9, 0.1]] [[0.1, 0.9]];
        [[0.75, 0.25]] [[0.25, 0.75]]
    ])(x, y)
    return z
end

# Get circuit for `test` in `PROPOSE` mode
# We must tell it the number of possible values that `input` can take (here, 2)
circuit = gen_fn_circuit(test, (input=2,), Propose())

smc_circuit(...)

### implement the circuit ###

REF_RATE() = 1.0
OFF_RATE() = 0.0001
ON_RATE() = 2.0 * REF_RATE()

Circuits.implement(::SIC.PositiveReal, ::Spiking) =
    SIC.SpikeRateReal(REF_RATE())
Circuits.implement(p::SIC.PositiveRealMultiplier, ::Spiking) =
    SIC.RateMultiplier(
        6.0, REF_RATE(),
        Tuple(SIC.SpikeRateReal(REF_RATE()) for _=1:p.n_inputs)
    )
Circuits.implement(c::SIC.CPTSampleScore, ::Spiking) =
    SIC.SpikingCPTSampleScore(c, OFF_RATE(), ON_RATE())

# implemented1 = implement(implement(circuit, Spiking()), Spiking())
# # implemented15 = implement(implement(implemented1.subcomponents[:sub_gen_fns], Spiking()), Spiking())
implemented2 = implement_deep(circuit, Spiking())

println("Component implemented.")

### visualize ###

# includet("../visualization/component_interface.jl")

# open("visualization/frontend/renders/partial_gen_fn.json", "w") do f
#     JSON.print(f, viz_graph(implemented15), 2)
# end
# println("Wrote component viz file.")

# # ### simulate ###

events = SpikingSimulator.simulate_for_time_and_get_events(implemented2, 20.0; initial_inputs=(:inputs => :input => 1,))

println("Simulation complete.")

# # ### spiketrain ###
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

using SpikingCircuits.SpiketrainViz

is_primary_output(compname, event) = (isnothing(compname) && event isa SpikingSimulator.OutputSpike)
dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))
for name in keys_deep(outputs(implemented2))
    if !haskey(dict, name)
        dict[name] = []
    end
end

draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0, xmax=20.
)

is_yval_spike((t, n, evt),) = (
       evt isa SpikingSimulator.OutputSpike
    && (
        n isa Pair && n.first === :sub_gen_fns
        && (
            (n.second isa Pair && contains(n.second.first, "y") && !(n.second.second isa Pair))
            || (!(n.second isa Pair) && contains(n.second, "y"))
        )
    )
    && evt.name isa Pair && evt.name.first === :value
)

is_z_inspike((t, n, evt),) = (
    evt isa SpikingSimulator.InputSpike
    && (
        (n isa Pair && n.first == :sub_gen_fns)
        && (
            (n.second isa Pair && contains(n.second.first, "z") && !(n.second.second isa Pair))
            || (!(n.second isa Pair) && contains(n.second, "z"))
        )
    )
)

# some util functions for inspecting the `events` array during debugging:
last(p::Pair) = p.second isa Pair ? last(p.second) : p.second
is_sgf_prob((t, n, evt),) = (
    evt isa SpikingSimulator.OutputSpike &&
    (
        n == :sub_gen_fns && evt.name isa Pair && last(evt.name) == :prob
    )
)
is_mult_prob_out((t, n, evt),) = (
    evt isa SpikingSimulator.OutputSpike &&
    (
        n == :multipliers
    )
)
is_mult_prob_in((t, n, evt),) = (
    evt isa SpikingSimulator.InputSpike &&
    (
        n == :multipliers
    )
)