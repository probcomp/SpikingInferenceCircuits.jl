"""
    ISParticle(model::GenFnWithInputDomains, proposal::GenFnWithInputDomains)

An importance sampling particle.
"""
struct ISParticle <: GenericComponent
    assess::GenFn{Generate}
    propose::GenFn{Propose}
end

ISParticle(model::GenFnWithInputDomains, proposal::GenFnWithInputDomains) =
    ISParticle(
        gen_fn_circuit(model, Assess()), gen_fn_circuit(proposal, Propose())
    )

Circuits.inputs(p::ISParticle) = NamedValues(
    :assess_args => inputs(p.assess)[:inputs],
    :obs => non_proposed_addresses(p),
    :propose_args => inputs(p.propose)[:inputs]
)

Circuits.outputs(p::ISParticle) = NamedValues(
    :trace => outputs(p.propose)[:trace],
    :weight => SingleNonnegativeReal()
)

function non_proposed_addresses(p)
    proposed_in = outputs(p.propose)[:trace] # CompositeValue
    inkeys = keys(proposed_in) # iterator over keys of some sort
    required_in = inputs(p.assess)[:obs] # CompositeValue
    reqkeys = keys(required_in)
    obsaddrs = setdiff(Set(reqkeys), Set(inkeys))
    return CompositeValue((;(
        addr => required_in[addr]
        for addr in obsaddrs
    )...))
end

propose_to_assess_edges(p) = (
    CompOut(:propose, :trace => addr) => CompIn(:assess, :obs => addr)
    for addr in keys(outputs(p.propose)[:trace])
)
obs_to_assess_edges(p) = (
    Input(:obs => addr) => CompIn(:assess, :obs => addr)
    for addr in keys(inputs(p)[:obs])
)

Circuits.implement(p::ISParticle, ::Target) =
    CompositeComponent(
        inputs(p), outputs(p), (
            assess=p.assess,
            propose=p.propose,
            multiplier=SDCs.NonnegativeRealMultiplier((
                outputs(p.assess)[:score],
                outputs(p.propose)[:score]
            ))
        ),
        (
            Input(:assess_args) => CompIn(:assess, :inputs),
            Input(:propose_args) => :CompIn(:propose, :inputs),
            propose_to_assess_edges(p)...,
            obs_to_assess_edges(p)...,
            CompOut(:assess, :score) => CompIn(:multiplier, 1),
            CompOut(:propose, :score) => CompIn(:multiplier, 2),
            CompOut(:propose, :trace) => Output(:trace),
            CompOut(:multiplier, :out) => Output(:weight)
        )
    )
