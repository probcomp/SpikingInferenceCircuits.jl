using ANNDistributions
using Distributions
using Flux
using Circuits, SpikingCircuits
Sim = SpikingCircuits.SpikingSimulator

# include("cpt_surrogate.jl")
# include("ann_cpt_sample.jl")

# # model = train_a_model(n_iters=30)
# import BSON
# # model = BSON.load("model-checkpoint.bson")[:model]
# model = BSON.load("simpler_model-checkpoint.bson")[:model]

# ann_on_assmt(assmt) = model(ANNDistributions.assmt_to_onehots(assmt, size(cpt)))
# # To continue training a model
# # model = train_a_model(;model, n_iters=30)
# # model = ANNDistributions.simpler_cpt_ann(cpt)
# # train_a_model(;model, n_iters=30)

# neuron_ΔT() = 100

# using SpikingInferenceCircuits
# const SIC = SpikingInferenceCircuits
# include("../../../experiments/utils/default_implementation_rules.jl")
# # TODO: deal with these parameters in a better way!
timer_params = (
    NSPIKES_SYNC_TIMER(),  #N_spikes_timer
    (1, M(), GATE_RATES()...), 0., # timer TI params (maxdelay M gaterates...) | offrate
    neuron_ΔT() * 10 # TODO: this should depend on n_layers!!!
)

# assmt = (4, 2, 1)
# truedist = cpt[assmt...]
# anndist = ann_on_assmt(assmt)

cptsample = ANNDistributions.ANNCPTSample(ANNDistributions.FullyConnectedANNWithDelay(ANNDistributions.FullyConnectedANN(model.layers |> collect, neuron_ΔT()), 50.,  timer_params), size(cpt))
#ANNCPTSample(model, neuron_ΔT(), 50., timer_params, size(cpt))
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
        println("no value found, so going to redo it!")
        return do_sim_get_val(impl, assmt)
    end
    val_evt = only(val_evts)
    return val_evt[3].name.second :: Int
end

#=
I think the current issue is that the initial layer forgets the inputs!
I think we need to add a layer of spikers before the first layer which just repeatedly feed in the inputted value.
=#