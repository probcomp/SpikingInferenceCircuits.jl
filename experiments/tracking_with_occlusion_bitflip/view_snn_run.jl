using Circuits, SpikingCircuits, Serialization
NLATENTS() = 5
include("../utils/spiketrain_utils.jl")

NPARTICLES() = 2
save_file() = "snn_runs/better_organized/rendered/2021-07-28__19-45"

events = (@time deserialize(save_file()));
inferred_states = get_smc_states(events, NPARTICLES(), NLATENTS())
