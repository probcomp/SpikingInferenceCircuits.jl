# This file produces runs for a grid of plots
# with proposal changing on the x axis
# and other parameters changing on the y axis

### Includes, ... ###
includet("../model/model.jl")
includet("../groundtruth_rendering.jl")
includet("../visualize.jl")
include("../proposals/obs_aux_proposal.jl")
includet("../proposals/prior_proposal.jl")
includet("../proposals/nearly_locally_optimal_proposal.jl")
includet("../run_utils.jl")

include("get_z_estimate_data.jl")

use_ngf() = false
if use_ngf()
    ProbEstimates.use_noisy_weights!()
else
    ProbEstimates.use_perfect_weights!()
end

@load_generated_functions()

### Do a particular run ###

ngf_setter(use_ngf, use_autonorm, use_aux_vars) =
    () -> begin
        (use_ngf ? ProbEstimates.use_noisy_weights! : ProbEstimates.use_perfect_weights!)()
        ProbEstimates.set_autonormalization!(use_autonorm)
        set_use_aux_vars!(use_aux_vars)
    end

specs = [
(step_prior_proposal, 10, ngf_setter(false, true, true)) (step_near_locopt_proposal, 10, ngf_setter(false, true, true)) ;
(step_prior_proposal, 10, ngf_setter(true, true, true)) (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
(step_prior_proposal, 10, ngf_setter(true, true, false)) (step_near_locopt_proposal, 10, ngf_setter(true, true, false)) ;
(step_prior_proposal, 10, ngf_setter(true, false, true)) (step_near_locopt_proposal, 10, ngf_setter(true, false, true)) ;
];

# We don't need these labels here, but we'll need them when we make the plot.
xlabels = ["Prior Proposal", "Nearly Locally\nOptimal Proposal"]
ylabels = ["Vanilla Gen", "NeuralGen-Fast\nw/ Auto-Norm\nw/ Aux Vars", "NeuralGen-Fast\nw/ Auto-Norm\nw/out Aux Vars", "NeuralGen-Fast\nw/out Auto-Norm\nw/ Aux vars"]

filename = run_and_save_z_estimates_comparison(
    [simulate(model, (15,)) for _=1:2],
    get_returned_obs,
    obs_choicemap_to_vec_of_vec,
    specs;
    n_particles_when_producing_prev_traces=2,
    n_particles_goldstandard=10,
    n_estimates_per_spec=4
);
println("Run saved to $filename.")