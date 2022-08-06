using Circuits, SpikingCircuits, Serialization

### Methods for visualizing a SNN run for the VelWalk1D model: ###

# fixed for this model
NLATENTS() = 1

# File with util to extract inferences from spiketrains
include("../utils/spiketrain_utils.jl")
# Include the file with the Gen model & visualization utils
include("run.jl")

function generate_tr(obs, xs, nsteps=(min(length(obs), length(xs)) - 1))
    (o0, x0), rest = Iterators.peel(zip(obs, xs))
    tr, _ = generate(model, (nsteps,), choicemap(
        (:init => :latents => :xₜ => :val, x0),
        (:init => :obs => :obs => :val, o0),
        Iterators.flatten(
            (
                (:steps => t => :latents => :xₜ => :val, x),
                (:steps => t => :obs => :obs => :val, o)
            )
            for (t, (o, x)) in enumerate(rest)
        )...
    ))
    return tr
end
# Currently the SMC circuit does not output coherent traces (ie. we only resample the last timestep,
# not the whole sequence which led to that).  So the traces we generate here will not be the
# traces resampled by SMC; all this guarantees is that the trace ending at time `t` has the inferred
# latents for a particle at time `t`.  This is all we need for the visualization to work.
_inferred_traces_for_time(obs, states) = [
    generate_tr(obs, [s[:xₜ][i] for s in states])
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
    obs, groundtruth_x
)
    inferred_states = get_smc_states(events, n_particles, NLATENTS())
    println("Obtained inferred states.")

    groundtruth_tr = generate_tr(obs, groundtruth_x, length(inferred_states) - 1);
    
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

save_file() = "snn_runs/better_organized/walk1d/fastgates/2021-07-28__22-05"

obs = [5, 8, 10, 17, 17, 15, 11, 13, 10, 10, 12]
groundtruth_x = [5, 8, 12, 12, 14, 13, 11, 13, 10, 9, 12]

# events = (@time deserialize(save_file()));

figure_for_smc_snn_run(
    events, NPARTICLES(), obs, groundtruth_x
)