"""
    ISParticle(
        proposal::ImplementableGenFn, latent_model::ImplementableGenFn,
        obs_model::ImplementableGenFn, latent_addr_order, obs_addr_order
    )

An importance sampling particle.  Given observations, samples latent variables
according to a proposal distribution, and outputs the proposed values and an unbiased
estimate of the importance weight.

The model may accept arguments ``\vec{x}``, and must decompose into
``P(\vec{l}, \vec{o} ; \vec{x}) = P_L(\vec{l} ; \vec{x})P_O(\vec{o} ; \vec{l})``
where ``\vec{l}`` is an assignment to the latent variables and ``\vec{o}`` is an assignment
to the observations.

The proposal must be a distribution ``Q(\vec{l} ; \vec{x}, \vec{o})``.

Arguments to the constructor:
- `proposal` is a ImplementableGenFn which accepts a sequence of arguments first giving
the model arguments``\vec{x}`` and then the observed values ``\vec{o}``; it traces values
at the addresses of the latent variables in the latent model.
- `latent_model` is a ImplementableGenFn which accepts a sequence of arguments ``\vec{x}``
and samples latent values ``\vec{l}``.
- `obs_model` is a ImplementableGenFn which accepts a sequence of latent values ``\tilde{l}``
which is a subset of all the traced latent values ``\vec{l}`` from the obs model.
It samples observation values ``\vec{o}``.
- `latent_addr_order` is a vector of addresses which are traced in the latent model.
It specifies the subset of the latent values ``\vec{l}`` which are passed to the
observation model (``\tilde{l}``), and the order in which they should be passed in.
- `obs_addr_order` is a vector of addresses which are traced in the obs model, giving the order
in which observations should be passed as inputs to the proposal (after regular arguments).

Restrictions:
- Say `assess_latents` has arguments with names `a1, ..., aN`.  
Then `propose` must have arguments with names `a1, ..., aN, o1, ..., oM`.
That is, propose gets all the same args as `assess_latents`, and then may get some additional arguments
which are observation values.  These value should be such that input
`o_i` receives the value traced at address `obs_addr_order[o_i]` in `assess_obs`.
- The `obs_model`'s inputs should be such that input `i` is traced at address
  `latent_addr_order[i]` in `assess_obs`.  Note that this means we currently do not support observation
  models which depend on arguments to the latent model (the dependencies must be a subset of the values
  traced in the latent model).
- The set of addresses traced in `propose` and `assess_latents` must be identical.
"""
struct ISParticle <: GenericComponent
    propose           :: GenFn{Propose}
    assess_latents    :: GenFn{Generate}
    assess_obs        :: GenFn{Generate}
    latent_addr_order :: Vector
    obs_addr_order    :: Vector
end
ISParticle(
    proposal::ImplementableGenFn, latent_model::ImplementableGenFn,
    obs_model::ImplementableGenFn, latent_addr_order, obs_addr_order
) = ISParticle(
    # Replace the return nodes since (1) the top-level return nodes don't matter to this circuit, and (2)
    # the return nodes may be tuples of values, which are currently not handled well by the compiler.
    # (They are treated as EnumeratedDomains spanning every possible assignment to the tuple, so there could be a huge
    # number of possible values.)
    # The return nodes will just be replaced with some node from within the IR.
    gen_fn_circuit(replace_return_node(proposal), Propose()), gen_fn_circuit(replace_return_node(latent_model), Assess()),
    gen_fn_circuit(replace_return_node(obs_model), Assess()), latent_addr_order, obs_addr_order
)

Circuits.inputs(p::ISParticle) = NamedValues(
    :args => inputs(p.assess_latents)[:inputs],
    :obs => inputs(p.assess_obs)[:obs]
)
Circuits.outputs(p::ISParticle) = NamedValues(
    :trace => outputs(p.propose)[:trace],
    :weight => SingleNonnegativeReal()
)

Circuits.implement(p::ISParticle, ::Target) =
    CompositeComponent(
        inputs(p), outputs(p), (
            propose=p.propose,
            assess_latents=p.assess_latents,
            assess_obs=p.assess_obs,
            multiplier=SDCs.NonnegativeRealMultiplier((
                outputs(p.propose)[:score],
                outputs(p.assess_latents)[:score],
                outputs(p.assess_obs)[:score]
            ))
        ), (
            ( # Input args -> propose args
                Input(:args => addr) => CompIn(:propose, :inputs => addr)
                for addr in keys(inputs(p.assess_latents)[:inputs])
                    if addr in keys(inputs(p.propose)[:inputs])
            )...,
            ( # Input obs -> propose args
                Input(:obs => addr) => CompIn(:propose, :inputs => addr)
                for addr in p.obs_addr_order
            )...,
            Input(:args) => CompIn(:assess_latents, :inputs),
            CompOut(:propose, :trace) => CompIn(:assess_latents, :obs),
            ( # sampled latent values -> assess obs args
                CompOut(:propose, :trace => prop_addr) => CompIn(:assess_obs, :inputs => obs_addr)
                for (prop_addr, obs_addr) in zip(
                    p.latent_addr_order, keys(inputs(p.assess_obs)[:inputs])
                )
            )...,
            Input(:obs) => CompIn(:assess_obs, :obs),
            CompOut(:propose, :score) => CompIn(:multiplier, 1),
            CompOut(:assess_latents, :score) => CompIn(:multiplier, 2),
            CompOut(:assess_obs, :score) => CompIn(:multiplier, 3),
            CompOut(:multiplier, :out) => Output(:weight),
            CompOut(:propose, :trace) => Output(:trace)
        ), p
    )