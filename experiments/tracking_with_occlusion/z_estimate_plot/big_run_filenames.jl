# I'm using this file to keep saved the filenames for some bigger SNN runs I did, for convenience.

# 3 Comparisons: 1 vs 10 vs 100 particles; Aux vars vs no aux vars; Prior proposal vs nearly_locally_optimal_proposal
# Git commit 2a74cd6f7355fd88cf92907851d635e36ad35406
particlecount_auxvar_proposal_20211007_1047AM = (
    filenames = [
        "/Users/georgematheos/Developer/research/Spiking/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/z_estimate_plot/saves/Particle Count2021-10-07__10-47-44",
        "/Users/georgematheos/Developer/research/Spiking/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/z_estimate_plot/saves/Uses Aux Vars2021-10-07__10-47-44",
        "/Users/georgematheos/Developer/research/Spiking/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/z_estimate_plot/saves/Proposal2021-10-07__10-47-44"
    ],
    specs = [
        (
            varied_quantity="Particle Count",
            constants_str="Near-Locally-Optimal Proposal\n NG-F w/ Auto-Normalization & Aux Vars",
            specs=[
                (step_near_locopt_proposal, 1, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 100, ngf_setter(true, true, true))
            ],
            labels=map(x -> "$x Particles", [1, 10, 100])
        ),
        (
            varied_quantity="Uses Aux Vars",
            constants_str="Near-Locally-Optimal Proposal\n NG-F w/ Auto-Normalization; 10 Particles",
            specs=[
                (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 10, ngf_setter(true, true, false)) ;
            ],
            labels=["with aux vars", "without aux vars"]
        ),
        (
            varied_quantity="Proposal",
            constants_str="NG-F w/ Auto-Normalization, w/ aux vars\n10 Particles",
            specs=[
                (step_prior_proposal, 10, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
            ],
            labels=["Prior as Proposal", "Nearly Locally\nOptimal Proposal"]
        ),
    ]
)

# ANN MAP in NG-F
proposal_comparison_ann_map_ngf_20211007_427pm = (
    filenames=["/Users/georgematheos/Developer/research/Spiking/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/z_estimate_plot/saves/Proposal2021-10-07__16-15-23"],
    specs=[
        (
            varied_quantity="Proposal",
            constants_str="NG-F w/ Auto-Normalization, w/ aux vars\n10 Particles",
            specs=[
                (step_prior_proposal, 10, ngf_setter(true, true, true)) ;
                (flux_proposal_MAP, 10, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
            ],
            labels=["Prior as Proposal", "Artificial Neural\nNetwork Proposal", "Nearly Locally\nOptimal Proposal"]
        ),
    ]
)

proposal_comparison_ann_map_ignore_ann_posoutput__ngf_20211007_459pm(
    filenames=["/Users/georgematheos/Developer/research/Spiking/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/z_estimate_plot/saves/Proposal2021-10-07__16-59-09"],
    specs = [
        (
            varied_quantity="Proposal",
            constants_str="NG-F w/ Auto-Normalization, w/ aux vars\n10 Particles",
            specs=[
                (step_prior_proposal, 10, ngf_setter(true, true, true)) ;
                (flux_proposal_MAP, 10, ngf_setter(true, true, true)) ;
                (step_near_locopt_proposal, 10, ngf_setter(true, true, true)) ;
            ],
            labels=["Prior as Proposal", "Artificial Neural\nNetwork Proposal", "Nearly Locally\nOptimal Proposal"]
        ),
    ]
)