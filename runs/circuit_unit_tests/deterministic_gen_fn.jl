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
        (500, 0.5, 50, 40)
        # Note that the R needs to be high since getting spikes while off is catastrophic.
        # TODO: Design things so this is not catastrophic (or can't happen at
        # realistic rates)!
    )

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
    SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()

XDOMAIN = 5
YDOMAIN = 2
ZDOMAIN = 4
fn = (x, y) -> mod(x + y, ZDOMAIN) + 1
circuit = gen_fn_circuit(fn, (FiniteDomain(XDOMAIN), FiniteDomain(YDOMAIN)), Propose())

println("Circuit constructed.")

shallow_implemented = implement(circuit, Spiking())

println("Circuit implemented.")

deep_implemented = implement_deep(circuit, Spiking())

println("Circuit implemented deeply.")

### simulation

include("spiketrain_utils.jl")

get_events(impl, x, y) = SpikingSimulator.simulate_for_time_and_get_events(impl, 500;
    initial_inputs=(:inputs => 1 => x, :inputs => 2 => y)
)

events = get_events(deep_implemented, 5, 2)
println("Simulation completed.")
draw_fig(events)
println("Figure drawn.")

