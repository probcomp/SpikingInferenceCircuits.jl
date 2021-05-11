#####
##### ISParticle
#####

struct ISParticle <: GenericComponent
    assess::GenFn{Generate} # really: GenFn{Assess}
    propose::GenFn{Propose}
end

function ISParticle(model::Gen.GenerativeFunction, 
        proposal::Gen.GenerativeFunction, 
        model_args_domain_sizes,
        proposal_args_domain_sizes)
    assess_circuit = gen_fn_circuit(model, model_args_domain_sizes, Assess())
    propose_circuit = gen_fn_circuit(proposal, proposal_args_domain_sizes, Propose())
    return ISParticle(assess_circuit, propose_circuit)
end

@doc(
"""
    struct ISParticle <: GenericComponent
        assess::GenFn{Generate}
        propose::GenFn{Propose}
    end

An `ISParticle` is a circuit component which represents an importance sampling particle (as would be used in inference algorithms like importance sampling or sequential Monte Carlo).
""", ISParticle)

#####
##### Circuits interface
#####

function Circuits.inputs(p::ISParticle)
    CompositeValue(
                   (assess_args = inputs(p.assess),
                    propose_args = inputs(p.propose)
                   )
                  )
end

function Circuits.outputs(p::ISParticle)
    CompositeValue(
                   (trace=outputs(p.propose)[:trace],
                    weight=NonnegativeReal()
                   )
                  )
end

function get_edges_propose_assess(p)
    (Pair(CompOut(:propose, :trace => addr), CompIn(:assess, :obs => addr))
     for addr in key(outputs(p.propose)[:trace]))
end

function Circuits.implement(p::ISParticle, t::Target)

    # Weights are represented internally by a tuple of Value instance.
    mult_unit = implement_deep(let w = score_value(p.assess)
                                   if w isa ProductNonnegativeReal
                                       NonnegativeRealMultiplier(w.factors)
                                   elseif w isa SingleNonnegativeReal
                                       NonnegativeRealMultiplier((w, ))
                                   end
                               end, t)

    return let full_in = implement_deep(inputs(p), t)
        full_out = implement_deep(outputs(p), t)
        CompositeComponent(full_in,
                           full_out,

                           # Subcomponents.
                           (  
                            assess=p.assess, # GenFn{Assess}
                            propose=p.propose, # GenFn{Propose}
                            multiplier=mult_unit,
                           ),

                           # Edges.
                           (
                            Iterators.flatten(
                                              (
                                               Pair(Input(:assess_args),
                                                    CompIn(:assess, :inputs)),
                                               Pair(Input(:propose_args),
                                                    CompIn(:propose, :inputs)),
                                               Pair(CompOut(:assess, :score),
                                                    CompIn(:multiplier))
                                               get_edges_propose_assess(p)...,
                                              )
                                             )...,
                           )
                          )
    end
end
