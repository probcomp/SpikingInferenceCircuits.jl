using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

### Implementation Rules ###
# TODO: a lot of the template of these rules
# could be written globally, with only
# setting the specific parameters done per-use.

Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
    SDCs.PulseToAssmts(
        ta, PulseIR.PoissonThresholdedIndicator,
        # ΔT, max_delay, M, R
        (500, 0.5, 50, 20)
        # Note that the R needs to be high since getting spikes while off is catastrophic.
        # TODO: Design things so this is not catastrophic (or can't happen at
        # realistic rates)!
    )

K = 10
ONRATE = 0.2
Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
    SDCs.PoissonPulseConditionalSample(
        (cs, K, ONRATE,
            500, # ΔT
            0.2, # max_delay
            1000, # M (num spikes to override offs/ons)
            50, # max delay before sample is emitted
            0.1 # intersample hold
        ),
        10^(-10), 12
    )

# TODO: ConditionalScore

### Tests ###

cpt = CPT([[0.5, 0.5], [0.2, 0.8]])
circuit = gen_fn_circuit(cpt, (FiniteDomain(2),), Propose())

println("Circuit constructed.")

shallow_implemented = implement(circuit, Spiking())

println("Circuit implemented.")

impl2 = implement(shallow_implemented, Spiking())

println("Implemented another level.")

impl3 = implement(impl2, Spiking())

println("Implemented another level.")

impl4 = implement(impl3, Spiking())

println("Implemented another level.")

deep_implemented = implement_deep(impl4, Spiking())

println("Circuit implemented deeply.")

# Simulation #
get_events() = SpikingSimulator.simulate_for_time_and_get_events(deep_implemented, 500;
    initial_inputs=(:inputs => 1 => 1,)
)

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

function draw_fig(events)
    dict = spiketrain_dict(
        filter(events) do (t, compname, event)
            (compname === :ss || compname === nothing) && event isa SpikingSimulator.OutputSpike
        end
    )
    draw_spiketrain_figure(
    collect(values(dict)); names=map(x->"$x", collect(keys(dict))), xmin=0
)
end

function simulate_and_log_to_file(filename)
    iostream = open(filename, "w")
    try
        SpikingSimulator.simulate_for_time(
            function (itr, time)
               for (name, evt) in itr
                    println(iostream, "$time | $name | $evt")
                    println("$time | $name | $evt")
               end
            end,
            #implement_deep(with_implemented_subcomponents.streamsamples, Spiking()),
            deep_implemented,
            500;
            initial_inputs=(:inputs => 1 => 1,),
            # initial_inputs=(1,),
            event_filter=((compname, event)->event isa SpikingSimulator.Spike)
        )
    catch e
    finally
        close(iostream)
    end
end

events = get_events()
println("Simulation completed.")
draw_fig(events)
println("Figure drawn")