using ANNDistributions
using Distributions
using Flux
using Circuits, SpikingCircuits
Sim = SpikingCircuits.SpikingSimulator

include("cpt_surrogate.jl")
include("ann_cpt_sample.jl")

# model = train_a_model(n_iters=30)
import BSON
model = BSON.load("model-checkpoint.bson")[:model]

ann_on_assmt(assmt) = model(ANNDistributions.assmt_to_onehots(assmt, size(cpt)))
# To continue training a model
# model = train_a_model(;model, n_iters=30)

neuron_ΔT() = 100

using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
include("../../../experiments/utils/default_implementation_rules.jl")
# TODO: deal with these parameters in a better way!
timer_params = (
    NSPIKES_SYNC_TIMER(),  #N_spikes_timer
    (1, M(), GATE_RATES()...), 0., # timer TI params (maxdelay M gaterates...) | offrate
    SYNC_TIMER_MEMORY() # timer memory
)

assmt = (4, 2, 1)
truedist = cpt[assmt...]
anndist = ann_on_assmt(assmt)

cptsample = ANNCPTSample(model, neuron_ΔT(), timer_params, size(cpt))
impl = implement_deep(cptsample, Spiking())
function do_sim_get_val(impl, assmt)
    events = Sim.simulate_for_time_and_get_events(
        impl, 400, initial_inputs=(
            :in_vals => varidx => varval
            for (varidx, varval) in enumerate(assmt)
        )
    )
    val_evts = filter(events) do (t, c, e); c === nothing && e.name isa Pair && e.name.first == :value; end
    if length(val_evts) < 1
        return do_sim_get_val(impl, assmts)
    end
    val_evt = only(val_evts)
    return val_evt[3].name.second :: Int
end