using Circuits, SpikingCircuits, Serialization

### Methods for visualizing a SNN run for the VelWalk1D model: ###

# fixed for this model
NLATENTS() = 5

# File with util to extract inferences from spiketrains
include("../utils/spiketrain_utils.jl")
# Include the file with the Gen model & visualization utils
include("run.jl")

function generate_tr(obs, poss, vels, nsteps=(min(length(obs), length(poss), length(vels)) - 1))
    (o0, p0, v0), rest = Iterators.peel(zip(obs, poss, vels))
    tr, _ = generate(model, (nsteps,), choicemap(
        (:init => :latents => :xₜ => :val, p0[1]),
        (:init => :latents => :yₜ => :val, p0[2]),
        (:init => :latents => :vₜ => :val, v0),
        (:init => :obs => :obsx => :val, o0[1]),
        (:init => :obs => :obsy => :val, o0[2]),
        Iterators.flatten(
            (
                (:steps => t => :latents => :xₜ => :val, x),
                (:steps => t => :latents => :yₜ => :val, y),
                (:steps => t => :latents => :vₜ => :val, v),
                (:steps => t => :obs => :obsx => :val, ox),
                (:steps => t => :obs => :obsy => :val, oy)
            )
            for (t, ((ox, oy), (x, y), v)) in enumerate(rest)
        )...
    ))
    return tr
end
vel_idx_to_label(idx) = idx + first(Vels()) - 1

# Currently the SMC circuit does not output coherent traces (ie. we only resample the last timestep,
# not the whole sequence which led to that).  So the traces we generate here will not be the
# traces resampled by SMC; all this guarantees is that the trace ending at time `t` has the inferred
# latents for a particle at time `t`.  This is all we need for the visualization to work.
# TODO:
_inferred_traces_for_time(obs, states) = [
   generate_tr(obs,
        [(s[:xₜ][i], s[:yₜ][i]) for s in states],
        [(vel_idx_to_label(s[:vxₜ][i]), vel_idx_to_label(s[:vyₜ][i])) for s in states]
    )
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
    obs, groundtruth_pos, groundtruth_vel
)
    inferred_states = get_smc_states(events, n_particles, NLATENTS())
    println("Obtained inferred states.")
    nsteps = length(inferred_states) - 1

    groundtruth_tr = generate_tr(obs, groundtruth_pos, groundtruth_vel, nsteps)
    
    return make_smc_figure(
        inferred_traces(obs, inferred_states), groundtruth_tr;
        n_particles,
        proposalstr="proposing from approx proposal IN SNN SIMULATION"
    )
end

### Visualize a particular run: ###
# save_file() = "snn_runs/better_organized/velwalk2d/2step/2021-07-24__15-52"
save_file() = "snn_runs/better_organized/velwalk2d/10step/2021-07-25__22-00"
NPARTICLES() = 2

obs = [(2, 1), (3, 3), (4, 6), (5, 7), (5, 10), (7, 10), (5, 8), (3, 6), (1, 3), (2, 3), (1, 2)]
groundtruth_pos = [(2, 1), (3, 3), (4, 5), (5, 7), (6, 9), (7, 10), (5, 8), (3, 6), (1, 4), (1, 3), (1, 2)]
groundtruth_vel = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (-2, -2), (-2, -2), (-2, -2), (-1, -1), (-1, -1)]

events = deserialize(save_file())
(fig, t) = figure_for_smc_snn_run(
    events, NPARTICLES(), obs, groundtruth_pos, groundtruth_vel
); fig