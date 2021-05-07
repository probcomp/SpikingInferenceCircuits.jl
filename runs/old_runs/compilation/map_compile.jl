using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

@gen (static) function inside(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input) # 2, 2
    return x
end

@gen (static) function add_minus_1(x, y)
    return x + y - 1
end
@gen (static) function outside(input)
    y1 ~ CPT([[0.9, 0.1], [0.1, 0.9]])(input) # 2
    y2 ~ CPT([[0.9, 0.1], [0.1, 0.9]])(input) # 2

    vec1 = [y1, 1]
    vec2 = [1, y2]
    
    sumvec ~ Map(add_minus_1)(vec1, vec2)

    output ~ Map(inside)(sumvec)

    return output
end
@load_generated_functions()

circuit = gen_fn_circuit(
    outside,
    (input=FiniteDomain(2),),
    Propose()
)

add_circuit = gen_fn_circuit(add_minus_1, (x=FiniteDomain(2), y=FiniteDomain(2)), Propose())

### implement the circuit ###

REF_RATE() = 0.5
OFF_RATE() = 0.0001
ON_RATE() = REF_RATE()

Circuits.implement(::SIC.PositiveReal, ::Spiking) =
    SIC.SpikeRateReal(REF_RATE())
Circuits.implement(p::SIC.PositiveRealMultiplier, ::Spiking) =
    SIC.RateMultiplier(
        6.0, REF_RATE(),
        Tuple(SIC.SpikeRateReal(REF_RATE()) for _=1:p.n_inputs)
    )
Circuits.implement(c::SIC.CPTSampleScore, ::Spiking) =
    SIC.SpikingCPTSampleScore(c, OFF_RATE(), ON_RATE())

implemented = implement_deep(circuit, Spiking())

println("Component implemented.")

events = SpikingSimulator.simulate_for_time_and_get_events(implemented, 100.0;
    initial_inputs=(:inputs => :input => 2,),
)

println("Simulation complete.")

### spiketrain ###
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
for name in keys_deep(outputs(implemented))
    if !haskey(dict, name)
        dict[name] = []
    end
end

draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0, xmax=100.
)

### Check spike rates: ###
probrate(dict, run_time) =
    if haskey(dict, :prob)
        length([x for x in dict[:prob] if x > 3/4 * run_time]) / (3/4 * run_time)
    else
        0.
    end
function do_run_and_check_spike_rate(run_length=100.0, input_val=2)
    events = SpikingSimulator.simulate_for_time_and_get_events(implemented, run_length;
        initial_inputs=(:inputs => :input => input_val,),
    )
    dict = spiketrain_dict(filter(((t,args...),) -> is_primary_output(args...), events))

    y1 = haskey(dict, :trace => :y1 => 1) ? 1 : 2
    y2 = haskey(dict, :trace => :y2 => 1) ? 1 : 2
    x1 = haskey(dict, :trace => :output => 1 => :x => 1) ? 1 : 2
    x2 = haskey(dict, :trace => :output => 2 => :x => 1) ? 1 : 2

    true_prob = exp(assess(outside, (input_val,), choicemap(
        (:y1, y1), (:y2, y2),
        (:output => 1 => :x, x1),
        (:output => 1 => :x, x2),
    ))[1])

    expected_rate = true_prob * REF_RATE()
    actual_rate   = probrate(dict, run_length)

    return (expected_rate, actual_rate)
end