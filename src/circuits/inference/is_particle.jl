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
    multiply_scores   :: Bool
end
ISParticle(
    proposal::ImplementableGenFn, latent_model::ImplementableGenFn,
    obs_model::ImplementableGenFn, latent_addr_order, obs_addr_order;
    multiply_scores         = true  :: Bool              ,
    truncate_proposal_dists = true  :: Bool              ,
    truncate_model_dists    = false :: Bool              ,
    truncation_minprob      = NaN   :: Float64
) = ISParticle(
    # Replace the return nodes since (1) the top-level return nodes don't matter to this circuit, and (2)
    # the return nodes may be tuples of values, which are currently not handled well by the compiler.
    # (They are treated as EnumeratedDomains spanning every possible assignment to the tuple, so there could be a huge
    # number of possible values.)
    # The return nodes will just be replaced with some node from within the IR.
    gen_fn_circuit(replace_return_node(proposal) |> maybe_truncate(truncate_proposal_dists, truncation_minprob), Propose()),
    gen_fn_circuit(replace_return_node(latent_model) |> maybe_truncate(truncate_model_dists, truncation_minprob), Assess()),
    gen_fn_circuit(replace_return_node(obs_model) |> maybe_truncate(truncate_model_dists, truncation_minprob), Assess()),
    latent_addr_order, obs_addr_order, multiply_scores
)
maybe_truncate(do_it, truncation_minprob) = do_it ? gf -> truncate_implementable_gf(gf, truncation_minprob) : identity

Circuits.inputs(p::ISParticle) = NamedValues(
    :args => inputs(p.assess_latents)[:inputs],
    :obs => inputs(p.assess_obs)[:obs]
)
Circuits.outputs(p::ISParticle) = NamedValues(
    :trace => outputs(p.propose)[:trace],
    :weight => weight_outval(p)
)
weight_outval(p) =
    if p.multiply_scores
        SingleNonnegativeReal()
    else
        ProductNonnegativeReal((
            outputs(p.propose)[:score],
            outputs(p.assess_latents)[:score],
            outputs(p.assess_obs)[:score]
        ))
    end
implemented_weight_outval(p, t) = p.multiply_scores ? weight_outval(p) : implement(weight_outval(p), t)

# no circuits if we don't multiply; a namedtuple with 1 multiplier if we do!
multiplier_circuits(p) =
    !p.multiply_scores ? () : (
                multiplier=SDCs.NonnegativeRealMultiplier((
                    outputs(p.propose)[:score],
                    outputs(p.assess_latents)[:score],
                    outputs(p.assess_obs)[:score]
                )),
            )

"""
An address in the trace to be fed into a proposal / recurred into a model as an argument,
along with a map saying how the sub-addresses of this address should be changed to sub-addresses
of the input value to the model.

E.g. in a trace with a 2D map leading to a sample from a `get_photon` address,
which we want to input into a simple `x => y` input value, we might use:
```julia
SIC.WithExtensionMap(
    :img_inner,
    trace_addr -> begin
        # trace addr will be
        # x => y => :got_photon
        (x, (y, rest)) = trace_addr
        @assert rest == :got_photon
        x => y # input to proposal using `x => y` [strip the `:got_photon`]
    end
```

Currently this is only implemented for obs values (ie. a `WithExtensionMap` can be
provided as an element of `obs_addr_order` in constructing an ISParticle [or SMC].)
We could add support for this in `latent_addr_order` too if needed, though ultimately
a better solution is https://github.com/probcomp/SpikingInferenceCircuits.jl/issues/23.
"""
struct WithExtensionMap
    addr
    map
end
edge_maybe_with_extension_map(fromval, from, to, addr::Union{<:Integer, Symbol}) =
    ( Circuits.append_to_valname(from, addr) => Circuits.append_to_valname(to, addr) , )
edge_maybe_with_extension_map(fromval, from, to, wem::WithExtensionMap) =
    (
        Circuits.append_to_valname(from, wem.addr => extension) =>
            Circuits.append_to_valname(to, wem.addr => wem.map(extension))
        for extension in keys_deep(fromval[wem.addr])
    )

Circuits.implement(p::ISParticle, t::Target) =
    CompositeComponent(
        inputs(p),
        NamedValues(
            :trace => outputs(p.propose)[:trace],
            :weight => implemented_weight_outval(p, t)
        ), 
        (
            propose=p.propose,
            assess_latents=p.assess_latents,
            assess_obs=p.assess_obs,
            multiplier_circuits(p)...
        ), (
            ( # Input args -> propose args
                Input(:args => addr) => CompIn(:propose, :inputs => addr)
                for addr in keys(inputs(p.assess_latents)[:inputs])
                    if addr in keys(inputs(p.propose)[:inputs])
            )...,
            Iterators.flatten( # Input obs -> propose args
                edge_maybe_with_extension_map(inputs(p)[:obs], Input(:obs), CompIn(:propose, :inputs), addr)
                for addr in p.obs_addr_order
            )..., # x => y => :got_photon         x => y
            Input(:args) => CompIn(:assess_latents, :inputs),
            CompOut(:propose, :trace) => CompIn(:assess_latents, :obs),
            ( # sampled latent values -> assess obs args
                CompOut(:propose, :trace => prop_addr) => CompIn(:assess_obs, :inputs => obs_addr)
                for (prop_addr, obs_addr) in zip(
                    p.latent_addr_order, keys(inputs(p.assess_obs)[:inputs])
                )
            )...,
            Input(:obs) => CompIn(:assess_obs, :obs),
            score_out_edges(p)...,
            CompOut(:propose, :trace) => Output(:trace)
        ), p
    )

score_out_edges(d) =
    if d.multiply_scores
        (
            CompOut(:propose, :score) => CompIn(:multiplier, 1),
            CompOut(:assess_latents, :score) => CompIn(:multiplier, 2),
            CompOut(:assess_obs, :score) => CompIn(:multiplier, 3),
            CompOut(:multiplier, :out) => Output(:weight)
        )
    else
        (
            CompOut(:propose, :score)        => Output(:weight => 1),
            CompOut(:assess_latents, :score) => Output(:weight => 2),
            CompOut(:assess_obs, :score)     => Output(:weight => 3),
        )
    end