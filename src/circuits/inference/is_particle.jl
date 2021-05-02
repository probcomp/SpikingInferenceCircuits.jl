struct ISParticle <: GenericComponent
    assess::GenFn{Generate} # really: GenFn{Assess}
    propose::GenFn{Propose}
    # ...
end

function ISParticle(
    model::Gen.GenerativeFunction,
    proposal::Gen.GenerativeFunction,
    model_args_domain_sizes,
    proposal_args_domain_sizes
)
    model_circuit = gen_fn_circuit(model, model_args_domain_sizes, Assess())
    propose_circuit = gen_fn_circuit(proposal, proposal_args_domain_sizes, Propose())

end

# CompositeValue(#= tuple or namedtuple =#)
# NamedValues(:a => Val(), :b => ...)
# IndexedValues(itr)

outputs(p.propose)[:trace]
# traceable_value(p.assess)
# iterator over ((addr, value)) pairs

Circuits.inputs(p::ISParticle) = CompositeValue((
    :obs => CompositeValue((; addr => FiniteDomainValue(n))),
    :propose_args => ,
    :assess_args => 
)
Circuits.outputs(p::ISParticle) = CompositeValue((
    trace=outputs(p.propose)[:trace],
    weight=PositiveReal()
))

Circuits.implement(p::ISParticle, ::Target) =
    CompositeComponent(
        inputs(p), outputs(p),
        ( # subcomponents
            assess=p.assess, # GenFn{Assess}
            propose=p.propose, # GenFn{Propose}
            multiplier=PositiveRealMultiplier(2)
        ),
        ( # Iterator over edges
            trace_to_assess(p)..., # maybe change to Iterators.flatten
            Input(:assess_args) => CompIn(:assess, :args),
            # other stuff ...
        )
    )

trace_to_assess(p) = (
    CompOut(:propose, :trace => addr) => CompIn(:assess, :obs => addr)
    for addr in key(outputs(p.propose)[:trace])
)

# (obs, propose_args, assess_args)

# maybe need an iterator to do this & distinguish the obs vs the proposal_args which are input to the propose subcomponent
(
    Input(:propose_args) => CompIn(:propose, :args)
)
(
    # obs into propose args
)

Input(:assess_args) => CompIn(:assess, :args)



Input(:obs) => CompIn(:propose, :trace => addr) for addr in keys(inputs(p.propose)[:trace])

# Iterator to do this
Input(:obs) => CompIn(:propose, )
 
=> Output()