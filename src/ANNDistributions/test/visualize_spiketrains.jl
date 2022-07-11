using Revise
using ANNDistributions
using Distributions
using Flux
using Circuits, SpikingCircuits
Sim = SpikingCircuits.SpikingSimulator

include("cpt_surrogate.jl")

# # model = train_a_model(n_iters=30)
import BSON

# model-checkpoint = 3 layer SNN with sizes 45->16->16->5
# simpler_model-checkpoint = 2 layer SNN with sizes 45->8->5
# model = BSON.load("model-checkpoint.bson")[:model]
model = BSON.load("simpler_model-checkpoint.bson")[:model]

ann_on_assmt(assmt) = model(ANNDistributions.assmt_to_onehots(assmt, size(cpt)))
# # To continue training a model
# model = train_a_model(;model, n_iters=30)
# # model = ANNDistributions.simple_cpt_ann(cpt)
# # train_a_model(;model, n_iters=30)

neuron_ΔT() = 10.

using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
include("../../../experiments/utils/default_implementation_rules.jl")
# # TODO: deal with these parameters in a better way!
timer_params = (
    NSPIKES_SYNC_TIMER(),  #N_spikes_timer
    (1, M(), GATE_RATES()...), 0. # timer TI params (maxdelay M gaterates...) | offrate
)

assmt = (20, 4, 19)
truedist = cpt[assmt...]
anndist = ann_on_assmt(assmt)

# cptsample = ANNDistributions.ANNCPTSample(ANNDistributions.FullyConnectedANNWithDelay(ANNDistributions.FullyConnectedANN(model.layers |> collect, neuron_ΔT()), 50.,  timer_params), size(cpt))
output_maxrate() = 50.
cptsample = ANNDistributions.ConcreteANNCPTSample(
    ANNCPTSample(model, size(cpt));
    neuron_memory=neuron_ΔT(), network_memory=100., timer_params,
    internal_maxrate=10.0, output_maxrate=output_maxrate()
)
@assert cdf(Poisson(output_maxrate() * MinProb() * neuron_ΔT()), RecipPEstDenom()) < 5e-5 "Too high a probability the ANN does not output a score!"

impl = implement_deep(cptsample, Spiking())

function get_val_score(events)
    val_evts = filter(events) do (t, c, e); c === nothing && e.name isa Pair && e.name.first == :value; end
    if length(val_evts) < 1
        @warn("no value found, so going to redo it!")
        return do_sim_get_val(impl, assmt)
    end
    if length(val_evts) > 1
        @error "Multiple val evts : $val_evts"
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
get_events(impl, assmt; simlen=100.) = Sim.simulate_for_time_and_get_events(
    impl, simlen, initial_inputs=(
        :in_vals => varidx => varval
        for (varidx, varval) in enumerate(assmt)
    ),
    log=true,
    log_filter=get_log_filter(400),
    log_str=time_log_str
)

counts_of_vals(vals) = Dict(
        val => length([v for v in vals if v == val])
        for val in unique(vals)
    )
dict_to_vector(d) = [
    get(d, i, 0) for i=1:maximum(keys(d))
]
extend_vector_to_length(v, l)= [i ≤ length(v) ? v[i] : 0 for i=1:l]

do_sim_get_val_score(args...; kwargs...) = get_val_score(get_events(args...; kwargs...))
do_sim_inspect_counts(args...; kwargs)   = inspect_counts(get_events(args...; kwargs...))

function do_sim_get_events_counts_val_score(args...; allow_no_score, kwargs...)
    events = get_events(args...; kwargs...)
    val_score = get_val_score(events)
    if !allow_no_score && isnothing(val_score[2])
        # if we got no score and this isn't allowed, try again!
        println("Redoing a run since we got no score from it!")
        return do_sim_get_counts_val_score(args...; allow_no_score, kwargs...)
    else
        return (events, (inspect_counts(events), val_score))
    end
end

using Printf: @sprintf
function get_log_filter(log_interval)
    cnt = 0
    function log_filter(time, compname, event)
        cnt += 1
        return (cnt - 1) % log_interval == 0
    end
    return log_filter
end
function time_log_str(time, compname, event)
    @sprintf("%.4f", time)
end

(events, counts_val_score) = do_sim_get_events_counts_val_score(impl, assmt; allow_no_score=false)
println("Simulation completed!")

println("Importing spiketrain visualizer...")
includet("../../../experiments/utils/spiketrain_utils.jl")
includet("../../../experiments/utils/spiketrain_utils/spiketrain_viz_utils.jl")

h = get_name_hierarchy(events)
labels = [
    (:ann => :input_layer => i for i in sort(collect(keys(h[:ann][:input_layer]))))...,
    (:ann => :ann => 1 => i for i in sort(collect(keys(h[:ann][:ann][1]))))...,
    (:ann => :ann => 2 => i for i in sort(collect(keys(h[:ann][:ann][2]))))...,
    :ann => :timer => :ti => :neuron, :ann => :timer => :neuron,
    :wta => :on => :neuron,
    (:wta => :offs => i => :neuron for i in sort(collect(keys(h[:wta][:offs]))))...,
    :counter => :ti => :neuron,
    (:counter => :mux => i => :component => :neuron for i in h[:counter][:mux] |> keys |> collect |> sort)...,
    :counter => :gate => :neuron
]
f = visualize_spiketrains_for_labels(labels, events)