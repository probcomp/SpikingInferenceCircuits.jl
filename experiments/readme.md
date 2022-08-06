# Contents of the `experiments` folder
This readme was last updated on August 6, 2022.

I don't fully understand the contents of this folder, as it
contains files from several git branches I worked on months apart, and then merged together
in August 2022.  But here is a list of pointers to the top-level files needed to produce
the figures for the SNMC paper.

- Tutorial model (1D object tracking) : `velwalk1d/` directory
    - Script to produce tutorial figure: `velwalk1d/figure/tutorial_figure.jl`
- Depth tracking model: `tracking_3d/` [[TODO: make note of the correct subdirectory]]
- Recursive concept learning : `number_game/`
    - Script to run this and save a visualization of the result: `number_game/run.jl`
- "Mental physics simulation" / image-likelihood model : `tracking_with_occlusion/` [[TODO: better understood the files in this directory]]

[[TODO what is `tracking_with_occlusion_bitflip`?]]
[[TODO understand + catalog the contents of the `pm_experiments` and `prob_estimate_tradeoffs` subdirectories]]