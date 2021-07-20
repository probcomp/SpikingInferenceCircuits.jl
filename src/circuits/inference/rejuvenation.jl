abstract type RejuvenationKernel <: GenericComponent end

# Rejuvenation kernel which does not change the latent values
# (using this means no rejuvenation occurs).
struct NoChangeRejuvKernel <: RejuvenationKernel
    args_input  :: CompositeValue
    obs_input   :: CompositeValue
    latents_val :: CompositeValue
end
Circuits.inputs(r::NoChangeRejuvKernel) = NamedValues(
    :prev_latents => r.latents_val,
    :obs => r.obs_input,
    :model_args => r.args_input
)
Circuits.outputs(r::NoChangeRejuvKernel) = NamedValues(:next_latents => r.latents_val)
Circuits.implement(r::NoChangeRejuvKernel, ::Target) = CompositeComponent(
    inputs(r), outputs(r), (), (Input(:prev_latents) => Output(:next_latents)), r
)

struct MHKernel <: RejuvenationKernel
    latent_model      ::GenFn{Generate} # Assess
    obs_model         ::GenFn{Generate} # Assess
    propose           ::GenFn{Propose}
    assess_proposal   ::GenFn{Generate} # Assess
    latent_addr_order :: Vector
    obs_addr_order    :: Vector
end
MHKernel(
    proposal::ImplementableGenFn,
    latent_model,
    obs_model,
    latent_addr_order, obs_addr_order
) =
    MHKernel(
        to_assess(latent_model),
        to_assess(obs_model),
        gen_fn_circuit(proposal, Propose()),
        gen_fn_circuit(proposal, Assess()),
        latent_addr_order, obs_addr_order
    )
to_assess(gf::GenFn{Generate}) = gf
to_assess(gf::ImplementableGenFn) = gen_fn_circuit(gf, Assess())

Circuits.inputs(mh::MHKernel) = NamedValues(
    :prev_latents => traceable_value(mh.latent_model),
    :model_args   => inputs(mh.latent_model)[:inputs],
    :obs          => inputs(mh.obs_model)[:obs]
)
Circuits.outputs(mh::MHKernel) = NamedValues(
    :next_latents => traceable_value(mh.latent_model)
)

Circuits.implement(mh::MHKernel, ::Target) =
    CompositeComponent(
        inputs(mh), outputs(mh),
        (
            propose=mh.propose,
            assess_bwd_proposal=mh.assess_proposal,
            assess_old_latents=mh.latent_model,
            assess_new_latents=mh.latent_model,
            assess_obs_old=mh.obs_model,
            assess_obs_new=mh.obs_model,
            new_trace_mult=NonnegativeRealMultiplier((
                outputs(mh.latent_model)[:score],
                outputs(mh.obs_model)[:score],
                outputs(mh.propose)[:score],
                outputs(mh.assess_proposal)[:score],
            )),
            old_trace_mult=NonnegativeRealMultiplier((outputs(mh.latent_model)[:score], outputs(mh.obs_model)[:score])),
            theta=Theta(2),
            latents_mux=Mux(2, traceable_value(mh.latent_model))
        ),
        (
            prev_latents_to_propose(mh)...,
            obs_to_propose(mh)...,

            Input(:model_args) => CompIn(:assess_new_latents, :inputs),

            edges_from_new_latents_to(mh, CompIn(:assess_new_latents, :obs))...,

            Input(:prev_latents) => CompIn(:assess_old_latents, :obs),
            Input(:model_args) => CompIn(:assess_old_latents, :inputs),

            prev_latents_to_assess_bwd(mh)...,
            new_latents_to_assess_bwd(mh)...,
            obs_to_assess_bwd(mh)...,
            
            CompOut(:assess_new_latents, :score) => CompIn(:new_trace_mult, 1),
            CompOut(:assess_obs_new, :score)     => CompIn(:new_trace_mult, 2),
            CompOut(:propose, :score)            => CompIn(:new_trace_mult, 3),
            CompOut(:assess_bwd_proposal, :score)    => CompIn(:new_trace_mult, 4),

            CompOut(:assess_old_latents, :score) => CompIn(:old_trace_mult, 1),
            CompOut(:assess_obs_old, :score)     => CompIn(:old_trace_mult, 2),

            CompOut(:old_trace_mult, :out) => CompIn(:theta, 1),
            CompOut(:new_trace_mult, :out) => CompIn(:theta, 2),

            CompOut(:theta, :val) => CompIn(:latents_mux, :sel),
            Input(:prev_latents) => CompIn(:latents_mux, :values => 1),
            edges_from_new_latents_to(mh, CompIn(:latents_mux, :values => 2))...,

            CompOut(:latents_mux, :out) => Output(:next_latents)
        ),
        mh
    )

prev_latents_to_propose(mh) = (
    Input(:prev_latents => latkey) => CompIn(:propose, :inputs => propkey)
    for (latkey, propkey) in zip(
        mh.latent_addr_order, keys(inputs(mh.propose)[:inputs])
    )
)
obs_to_propose(mh) = (
    Input(:obs => obskey) => CompIn(:propose, :inputs => propkey)
    for (obskey, propkey) in zip(
        mh.obs_addr_order,
        Iterators.drop(keys(inputs(mh.propose)[:inputs]), length(mh.latent_addr_order))
    )
)
prev_latents_to_assess_bwd(mh) = (
    Input(:prev_latents => key) => CompIn(:assess_bwd_proposal, :obs => key)
    for key in keys(inputs(mh.assess_proposal)[:obs])
)

# Automatically use the previous latent values for anything that isn't proposed!
new_latent_val(mh, key) =
    haskey(outputs(mh.propose)[:trace], key) ?
        CompOut(:propose, :trace => key)     :
        Input(:prev_latents => key)

edges_from_new_latents_to(mh, receiver) = (
    new_latent_val(mh, key) => Circuits.append_to_valname(receiver, key)
    for key in keys(traceable_value(mh.latent_model))
)

new_latents_to_assess_bwd(mh) = (
    new_latent_val(mh, tracekey) => CompIn(:assess_bwd_proposal, :inputs => propkey)
    for (propkey, tracekey) in zip(
        keys(inputs(mh.assess_proposal)[:inputs]),
        mh.latent_addr_order
    )
)
obs_to_assess_bwd(mh) = (
    Input(:obs => obskey) => CompIn(:assess_bwd_proposal, :inputs => propkey)
    for (propkey, obskey) in zip(
        Iterators.drop(keys(inputs(mh.assess_proposal)[:obs]), length(mh.latent_addr_order)),
        mh.obs_addr_order
    )
)
