using ANNDistributions
using Distributions
using Flux
using Circuits, SpikingCircuits
Sim = SpikingCircuits.SpikingSimulator

include("cpt_surrogate.jl")
include("ann_cpt_sample.jl")

# # model = train_a_model(n_iters=30)
import BSON

# model-checkpoint = 3 layer SNN with sizes 45->16->16->5
# simpler_model-checkpoint = 2 layer SNN with sizes 45->8->5
# model = BSON.load("model-checkpoint.bson")[:model]
# model = BSON.load("simpler_model-checkpoint.bson")[:model]

ann_on_assmt(assmt) = model(ANNDistributions.assmt_to_onehots(assmt, size(cpt)))
# # To continue training a model
# model = train_a_model(;model, n_iters=30)
# # model = ANNDistributions.simple_cpt_ann(cpt)
# # train_a_model(;model, n_iters=30)

neuron_ΔT() = 100

using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
include("../../../experiments/utils/default_implementation_rules.jl")
# # TODO: deal with these parameters in a better way!
timer_params = (
    NSPIKES_SYNC_TIMER(),  #N_spikes_timer
    (1, M(), GATE_RATES()...), 0., # timer TI params (maxdelay M gaterates...) | offrate
    neuron_ΔT() * 10 # TODO: this should depend on n_layers!!!
)

assmt = (20, 4, 19)
truedist = cpt[assmt...]
anndist = ann_on_assmt(assmt)

# cptsample = ANNDistributions.ANNCPTSample(ANNDistributions.FullyConnectedANNWithDelay(ANNDistributions.FullyConnectedANN(model.layers |> collect, neuron_ΔT()), 50.,  timer_params), size(cpt))
cptsample = ANNCPTSample(model, neuron_ΔT(), 1000., timer_params, size(cpt))
impl = implement_deep(cptsample, Spiking())

function get_val_score(events)
    val_evts = filter(events) do (t, c, e); c === nothing && e.name isa Pair && e.name.first == :value; end
    if length(val_evts) < 1
        println("no value found, so going to redo it!")
        return do_sim_get_val(impl, assmt)
    end
    val_evt = only(val_evts)

    score_evt_names = [e.name for (t, c, e) in events if c === nothing && e.name isa Pair && e.name.first == :inverse_prob]
    indspikes = [name for name in score_evt_names if name.second == :ind]
    @assert length(indspikes) ≤ 1
    got_ind = length(indspikes) > 0
    score = got_ind ? (length(score_evt_names) - 1) / RecipPEstDenom() : nothing

    return (val_evt[3].name.second :: Int, score)
end

function inspect_counts(events)
    timertimes = [t for (t, c, e) in events if c == (:ann => :timer) && e isa Sim.OutputSpike]
    time = only(timertimes)
    outspikes_idxs = [e.name for (t, c, e) in events if c == :ann && e isa Sim.OutputSpike]
    unique_idxs = unique(outspikes_idxs)
    return Dict(
        idx => length([i for i in outspikes_idxs if i == idx])
        for idx in unique_idxs
    )

end
get_events(impl, assmt; simlen=400.) = Sim.simulate_for_time_and_get_events(
    impl, simlen, initial_inputs=(
        :in_vals => varidx => varval
        for (varidx, varval) in enumerate(assmt)
    )
)

counts_of_vals(vals) = Dict(
        val => length([v for v in vals if v == val])
        for val in unique(vals)
    )
dict_to_vector(d) = [
    get(d, i, 0) for i=1:maximum(keys(d))
]

do_sim_get_val_score(args...; kwargs...) = get_val_score(get_events(args...; kwargs...))
do_sim_inspect_counts(args...; kwargs)   = inspect_counts(get_events(args...; kwargs...))
function do_sim_get_counts_val_score(args...; kwargs...)
    events = get_events(args...; kwargs...)
    return (inspect_counts(events), get_val_score(events))
end