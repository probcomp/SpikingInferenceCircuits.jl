using Circuits, SpikingCircuits, Serialization

### Methods for visualizing a SNN run for the VelWalk1D model: ###

# fixed for this model
NLATENTS() = 2

# File with util to extract inferences from spiketrains
include("../utils/spiketrain_utils.jl")
# Include the file with the Gen model & visualization utils
include("run.jl")

function generate_tr(obs, xs, vs, nsteps=(min(length(obs), length(xs), length(vs)) - 1))
    (o0, x0, v0), rest = Iterators.peel(zip(obs, xs, vs))
    tr, _ = generate(model, (nsteps,), choicemap(
        (:init => :latents => :xₜ => :val, x0),
        (:init => :latents => :vₜ => :val, v0),
        (:init => :obs => :obs => :val, o0),
        Iterators.flatten(
            (
                (:steps => t => :latents => :xₜ => :val, x),
                (:steps => t => :latents => :vₜ => :val, v),
                (:steps => t => :obs => :obs => :val, o)
            )
            for (t, (o, x, v)) in enumerate(rest)
        )...
    ))
    return tr
end

vel_idx_to_label(idx) = idx + first(Vels()) - 1
# Currently the SMC circuit does not output coherent traces (ie. we only resample the last timestep,
# not the whole sequence which led to that).  So the traces we generate here will not be the
# traces resampled by SMC; all this guarantees is that the trace ending at time `t` has the inferred
# latents for a particle at time `t`.  This is all we need for the visualization to work.
_inferred_traces_for_time(obs, states) = [
    generate_tr(obs, [s[:xₜ][i] for s in states], [vel_idx_to_label(s[:vₜ][i]) for s in states])
    for i=1:NPARTICLES()
]
_inferred_traces(obs, states) = [
    _inferred_traces_for_time(obs, @view(states[1:i]))
    for i=1:length(states)
]
inferred_traces(obs, states) = (_, _) -> (_inferred_traces(obs, states), nothing)

figure_for_smc_snn_run(snn_events_filename::String, args...) = figure_for_smc_snn_run(deserialize(snn_events_filename),  args...)
function figure_for_smc_snn_run(
    events::Vector, n_particles,
    obs, groundtruth_x, groundtruth_v
)
    inferred_states = get_smc_states(events, n_particles, NLATENTS())
    println("Obtained inferred states.")

    groundtruth_tr = generate_tr(obs, groundtruth_x, groundtruth_v, length(inferred_states) - 1);
    
    println(get_args(last(inferred_traces(obs, inferred_states)(nothing, nothing)[1])[1]))
    println(get_args(groundtruth_tr))
    return make_smc_figure(
        inferred_traces(obs, inferred_states), groundtruth_tr;
        n_particles,
        proposalstr="proposing from approx proposal IN SNN SIMULATION"
    )
end

### Script to create the figure for a particular run: ###
NPARTICLES() = 2
# save_file() = "snn_runs/better_organized/velwalk1d/2timesteps2000interval/2021-07-19__18-23"
# save_file() = "snn_runs/better_organized/velwalk1d/10timesteps2000interval/2021-07-20__02-42"
save_file() = "snn_runs/better_organized/velwalk1d/10timesteps200interval/2021-07-26__02-02"
# save_file() = "snn_runs/better_organized/velwalk1d_pm/2timesteps2000interval/2021-07-20__18-46"
# save_file() = "snn_runs/better_organized/velwalk1d_pm/10timesteps2000interval--failed/2021-07-21__01-30"
# save_file() = "snn_runs/better_organized/velwalk1d_mh/2timesteps2000interval/2021-07-20__18-58"

# save_file() = "snn_runs/better_organized/velwalk1d/slow_gates/2021-07-26__19-58"
# save_file() = "snn_runs/better_organized/velwalk1d_pm/10timesteps2000interval/2021-07-26__02-08"
# save_file() = "snn_runs/better_organized/velwalk1d/slow_gates/10step/2021-07-27__18-43"
# save_file() = "snn_runs/better_organized/velwalk1d_ann/2021-07-27__18-23"

obs           = [16, 15, 11, 10, 8, 14, 11, 17, 18, 18, 18]
groundtruth_x = [16, 14, 12, 10, 8, 10, 12, 14, 16, 19, 20]
groundtruth_v = [-2, -2, -2, -2, -2, 2, 2, 2, 2, 3, 1]

events = (@time deserialize(save_file()));

figure_for_smc_snn_run(
    events, NPARTICLES(), obs, groundtruth_x, groundtruth_v
)