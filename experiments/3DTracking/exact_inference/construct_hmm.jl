using GenDiscreteHMM

@time hmm = labeled_discrete_hmm_from_generative_functions(
    [ # Iterator over every dynamic latent state
        # choicemap(
        #     (:vxₜ, vx), (:vyₜ, vy), (:vzₜ, vz),
        #     (:xₜ, x), (:yₜ, y), (:zₜ, z)
        # )
        StaticChoiceMap((; vxₜ=vx, vyₜ=vy, vzₜ=vz, xₜ=x, yₜ=y, zₜ=z), (;))
        for vx in Vels() for vy in Vels() for vz in Vels() for x in Xs() for y in Ys() for z in Zs()
    ],
    [ # Iterator over every transient latent state
        # choicemap(
        #     (:true_ϕ, ϕ), (:true_θ, θ), (:rₜ, r)
        # )
        StaticChoiceMap((; true_ϕ=ϕ, true_θ=θ), (;))
        for ϕ in ϕs() for θ in θs() for r in Rs()
    ],
    # generative functions
    initial_model, transient_state_model, step_model, obs_model;
    # function specifying how to use the latent (x) and transient (u) choicemaps
    # to get the arguments to the generative functions:
    transient_state_model_arg_format=(
        x -> (x[:xₜ], x[:yₜ], x[:zₜ])
    ),
    step_model_arg_format=(
        x -> (x[:vxₜ], x[:vyₜ], x[:vzₜ], x[:xₜ], x[:yₜ], x[:zₜ])
    ),
    likelihood_arg_format=(
        (u, _) -> (u[:true_ϕ], u[:true_θ])
    )
);

GenDiscreteHMM.serialize_hmm_contents("medium_hmm_contents.jld", hmm);