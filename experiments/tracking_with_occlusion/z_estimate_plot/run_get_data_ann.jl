# This file produces runs for a comparison between proposals, including the ANN proposal.

### Includes, ... ###
includet("../model/model.jl")
includet("../groundtruth_rendering.jl")
includet("../visualize.jl")
include("../proposals/obs_aux_proposal.jl")
includet("../proposals/prior_proposal.jl")
includet("../proposals/nearly_locally_optimal_proposal.jl")
includet("../proposals/ann_proposal.jl")
includet("../run_utils.jl")

include("get_z_estimate_data.jl")

# Make sure the probabilities in the model/proposal are not being truncated!
# And make sure to set the minprob threshold low enough that this isn't a problem.
ProbEstimates.TruncateRecipDists() = ProbEstimates.weight_type() !== :perfect
ProbEstimates.TruncateFwdDists() = false
ProbEstimates.DoRecipPECheck() = true
ProbEstimates.set_assembly_size!(25)

use_ngf() = false
if use_ngf()
    ProbEstimates.use_noisy_weights!()
else
    ProbEstimates.use_perfect_weights!()
end

@load_generated_functions()

### Run specs ###

min_prob_auxvars = min(prob_flip1_given_no_flip(), √(ColorFlipProb()))
min_prob_noaux   = ColorFlipProb()
ngf_setter(use_ngf, use_autonorm, use_aux_vars) =
    () -> begin
        (use_ngf ? ProbEstimates.use_noisy_weights! : ProbEstimates.use_perfect_weights!)()
        ProbEstimates.set_autonormalization!(use_autonorm)
        set_use_aux_vars!(use_aux_vars)
        ProbEstimates.set_minprob!(use_aux_vars ? min_prob_auxvars : min_prob_noaux)
        println("K recip: $(ProbEstimates.K_recip())")
    end

generate_filenames(specs) =
    let date = Dates.now()
        [
            generate_filename(; date, filename=varied_quantity)
            for (varied_quantity, _...) in specs
        ]
    end

specs = [
    (
        varied_quantity="Proposal",
        constants_str="NG-F w/ Auto-Normalization, w/ aux vars\n10 Particles",
        specs=[
            (step_prior_proposal, 10, ngf_setter(true, true, true)) ;
            (flux_proposal_MAP, 10, ngf_setter(true, true, true)) ;
            (flux_proposal, 10, ngf_setter(true, true, true)) ;
            (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
        ],
        labels=["Prior as Proposal", "Deterministic ANN Proposal", "Stochastic ANN Proposal", "Nearly Locally\nOptimal Proposal"]
    ),
]

filenames = generate_filenames(specs)
for (filename, spec) in zip(filenames, specs)
    println("Doing run which varies $(spec.varied_quantity)...")
    # name = run_and_save_z_estimates_comparison(
    #     [simulate(model, (15,)) for _=1:5],
    #     get_returned_obs,
    #     obs_choicemap_to_vec_of_vec,
    #     spec.specs;
    #     n_particles_when_producing_prev_traces=8,
    #     n_particles_goldstandard=20,
    #     n_estimates_per_spec=4,
    #     filename
    # );
    name = run_and_save_z_estimates_comparison(
        [simulate(model, (15,)) for _=1:2],
        get_returned_obs,
        obs_choicemap_to_vec_of_vec,
        spec.specs;
        n_particles_when_producing_prev_traces=4,
        n_particles_goldstandard=10,
        n_estimates_per_spec=1,
        filename
    );
    println("Run saved to $name.")
end