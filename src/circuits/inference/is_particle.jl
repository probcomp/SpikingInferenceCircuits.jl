#####
##### ISParticle
#####

struct ISParticle <: GenericComponent
    assess::GenFn{Generate}
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
                    weight=SingleNonnegativeReal()
                   )
                  )
end

# Creates a Generator for edges between propose and assess sub-circuits.
function get_edges_propose_assess(p)
    (Pair(CompOut(:propose, :trace => addr), 
          CompIn(:assess, :obs => addr))
     for addr in keys(outputs(p.propose)[:trace]))
end

function Circuits.implement(p::ISParticle, t::Target)

    # Weights are represented internally by a tuple of Value instance.
    mult_unit = SDCs.NonnegativeRealMultiplier((
                                        outputs(p.assess)[:score],
                                        outputs(p.propose)[:score]
                                               ))

    return CompositeComponent(inputs(p),
                              outputs(p),

                              # Subcomponents.
                              (  
                               assess=p.assess, # GenFn{Generate}
                               propose=p.propose, # GenFn{Propose}
                               multiplier=mult_unit,
                              ),

                              # Edges.
                              (
                               Pair(Input(:assess_args => :inputs),
                                    CompIn(:assess, :inputs)),
                               Pair(Input(:propose_args => :inputs),
                                    CompIn(:propose, :inputs)),
                               get_edges_propose_assess(p)...,
                               Pair(CompOut(:propose, :trace), 
                                    Output(:trace)),
                               Pair(CompOut(:multiplier, :out),
                                    Output(:weight))
                              )
                             )
end
