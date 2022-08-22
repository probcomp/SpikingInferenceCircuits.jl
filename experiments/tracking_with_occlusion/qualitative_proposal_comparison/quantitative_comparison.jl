includet("utils.jl")

#=
Generate a trace where
1. at timestep 0, at (2, 5) with vel (2, 0); occ at 5
2. at timestep 1, at (4, 5) with vel (2, 0); occ at 5
3. at timestep 2, at (6, 5) with vel (2, 0); occ at 5 (so now it's behind the occluder)

Inference will be done over timestep 2.
=#
ColorFlipProb() = 0.0
informative_dynamics_uninformative_obs_tr, _ = generate(model_noflipvars, (2,), choicemap(
    (:init => :latents => :occₜ => :val, 5),
    (:init => :latents => :xₜ => :val, 2),
    (:init => :latents => :yₜ => :val, 5),
    (:init => :latents => :vxₜ => :val, 2),
    (:init => :latents => :vyₜ => :val, 0),
    
    (:steps => 1 => :latents => :occₜ => :val, 5),
    (:steps => 1 => :latents => :xₜ => :val, 4),
    (:steps => 1 => :latents => :yₜ => :val, 5),
    (:steps => 1 => :latents => :vxₜ => :val, 2),
    (:steps => 1 => :latents => :vyₜ => :val, 0),

    (:steps => 2 => :latents => :occₜ => :val, 5),
    (:steps => 2 => :latents => :xₜ => :val, 6),
    (:steps => 2 => :latents => :yₜ => :val, 5),
    (:steps => 2 => :latents => :vxₜ => :val, 2),
    (:steps => 2 => :latents => :vyₜ => :val, 0),
))
informative_dynamics_informative_obs_tr, _ = 
    generate(model_noflipvars, (2,), choicemap(
        (:init => :latents => :occₜ => :val, 5),
        (:init => :latents => :xₜ => :val, 2),
        (:init => :latents => :yₜ => :val, 5),
        (:init => :latents => :vxₜ => :val, 1),
        (:init => :latents => :vyₜ => :val, 0),
        
        (:steps => 1 => :latents => :occₜ => :val, 5),
        (:steps => 1 => :latents => :xₜ => :val, 3),
        (:steps => 1 => :latents => :yₜ => :val, 5),
        (:steps => 1 => :latents => :vxₜ => :val, 1),
        (:steps => 1 => :latents => :vyₜ => :val, 0),

        (:steps => 2 => :latents => :occₜ => :val, 5),
        (:steps => 2 => :latents => :xₜ => :val, 4),
        (:steps => 2 => :latents => :yₜ => :val, 5),
        (:steps => 2 => :latents => :vxₜ => :val, 1),
        (:steps => 2 => :latents => :vyₜ => :val, 0),
    ))
surprising_dynamics_informative_obs_tr, _ = 
    generate(model_noflipvars, (2,), choicemap(
        (:init => :latents => :occₜ => :val, 5),
        (:init => :latents => :xₜ => :val, 2),
        (:init => :latents => :yₜ => :val, 5),
        (:init => :latents => :vxₜ => :val, 1),
        (:init => :latents => :vyₜ => :val, 0),
        
        (:steps => 1 => :latents => :occₜ => :val, 5),
        (:steps => 1 => :latents => :xₜ => :val, 3),
        (:steps => 1 => :latents => :yₜ => :val, 5),
        (:steps => 1 => :latents => :vxₜ => :val, 1),
        (:steps => 1 => :latents => :vyₜ => :val, 0),

        (:steps => 2 => :latents => :occₜ => :val, 5),
        (:steps => 2 => :latents => :xₜ => :val, 3),
        (:steps => 2 => :latents => :yₜ => :val, 6),
        (:steps => 2 => :latents => :vxₜ => :val, -1),
        (:steps => 2 => :latents => :vyₜ => :val, 1),
    ))
ColorFlipProb() = 0.01


f, trips = make_plots_with_visuals(
    [
        ["Predictable Dynamics; Uninformative Observation", informative_dynamics_uninformative_obs_tr],
        ["Predictable Dynamics; Informative Observation", informative_dynamics_informative_obs_tr],
        ["Surprising Dynamics; Informative Observation", surprising_dynamics_informative_obs_tr]
    ]
); f

#=
TODOs:
[x] 5th and 95th percentiles
[x] Legend for visual
[x] Change arrow to show most continuaton along previous velocity (rather than showing motion along new velocity)
[- Change layout so each row has (1) scenario, (2) plot, (3) sequence of panels showing heatmap]
=#