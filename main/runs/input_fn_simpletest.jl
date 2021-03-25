import JSON

using Circuits
using SpikingCircuits
const Sim = SpikingSimulator
using Distributions: Categorical

# events = Sim.simulate_for_time_and_get_events(
#     InputFunctionPoisson((identity,), (10.0,), x -> 1 + x),
#     20.0;
#     initial_inputs=(1, 1)
# )

# spikes = filter(events) do (t, name, evt)
#     evt isa Sim.OutputSpike
# end

neuron = InputFunctionPoisson((identity,), (10.0,), x -> (1 + x)/4)
st = SpikingCircuits.InputTimesState([[9.0]])
xtnd() = Sim.extend_trajectory(neuron, st, Sim.EmptyTrajectory())
[xtnd() for _=1:100]