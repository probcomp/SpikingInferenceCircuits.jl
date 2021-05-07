"""
    CPTScore <: GenericComponent

Unit to output `P[value ; parent_values]` for a CPT.
"""
struct CPTScore <: GenericComponent
    cpt::CPT
end

Circuits.inputs(c::CPTScore) = NamedValues(
    :in_vals => IndexedValues(
        FiniteDomainValue(n)
        for n in input_ncategories(c.cpt)
    ),
    :obs => FiniteDomainValue(ncategories(c.cpt))
)
Circuits.outputs(::CPTScore) = NamedValues(:prob => PositiveReal())

Circuits.implement(c::CPTScore, ::Target) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            assmts=ToAssmts(input_ncategories(c.cpt)),
            score=ConditionalScore(assmt_cond_prob_matrix(c.cpt))
        ),
        (
            (
                Input(:in_vals => i) => CompIn(:assmts, i)
                for i=1:num_inputs(c.cpt)
            )...,
            Input(:obs) => CompIn(:score, :obs),
            CompOut(:assmts, :out) => CompIn(:score, :in_val),
            CompOut(:score, :prob) => Output(:prob)
        ),
        c
    )