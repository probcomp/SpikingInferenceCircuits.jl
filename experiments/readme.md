# Contents of the `experiments` folder
This readme was last updated on August 6, 2022.

I don't fully understand the contents of this folder, as it
contains files from several git branches I worked on months apart, and then merged together
in August 2022.  But here is a list of pointers to the top-level files needed to produce
the figures for the SNMC paper.

- Tutorial model (1D object tracking) : `velwalk1d/` directory
    - Script to produce tutorial figure: `velwalk1d/figure/tutorial_figure.jl`
- Recursive concept learning : `number_game/`
    - Script to run this and save a visualization of the result: `number_game/run.jl`
- Fractional-variance scaling plot : `tracking_with_occlusion/scaling_plots/main.jl`
- "Mental physics simulation" / "Image-likelihood tracking" / "Tracking with occlusion" inference runs : `tracking_with_occlusion/main.jl`
- "Mental physics simulation" (etc.) qualitative proposal distribution comparison : `tracking_with_occlusion/qualitative_proposal_comparison/main.jl`

[[
merge TODOs:
- 3Dtracking
- 1dvel to GenSN-Sim
- what is `prob_estimate_tradeoffs` ?
- what is `pm_experiments` ?
- what is `tracking_with_occlusion_bitflip`
- how to make GenSN-Em spiketrain renders?
- somehow document `setup.jl` (or general documentation)?

fig TODOs:
- understand what noise model is used in `tracking_with_occlusion/main.jl`
]]