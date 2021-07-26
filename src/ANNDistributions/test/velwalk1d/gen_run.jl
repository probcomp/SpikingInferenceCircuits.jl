include("setup.jl")

smc_ann_proposal(tr, n_particles) = smc(tr, n_particles, exact_init_proposal, ann_step_proposal)

include("../../../../experiments/velwalk1d/run_utils.jl")

make_smcann_2d_posterior_figure(tr; n_particles=10) =
    make_smc_figure(smc_ann_proposal, tr; n_particles, proposalstr="proposing using ANN")

tr, _ = generate(model, (10,));
make_smcann_2d_posterior_figure(tr)