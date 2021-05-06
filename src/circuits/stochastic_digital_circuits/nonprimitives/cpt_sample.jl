"""
    CPTSample <: GenericComponent

Unit to sample from a CPT and output the sample and `1/P[sample ; inputs]`.
"""
struct CPTSample <: GenericComponent
    cpt::CPT
end

Circuits.inputs(c::CPTSample) = NamedValues(
    :in_vals => IndexedValues(
        FiniteDomainValue(n)
        for n in input_ncategories(c.cpt)
    )
)
Circuits.outputs(c::CPTSample) =
    NamedValues(
        :value => FiniteDomainValue(ncategories(c.cpt)),
        :inverse_prob => PositiveReal()
    )

Circuits.implement(c::CPTSample, ::Target) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            assmts=ToAssmts(input_ncategories(c.cpt)),
            sample=ConditionalSample(assmt_cond_prob_matrix(c.cpt))
        ),
        (
            (
                Input(:in_vals => i) => CompIn(:assmts, i)
                for i=1:num_inputs(c.cpt)
            )...,
            CompOut(:assmts, :out) => CompIn(:sample, :in_val),
            CompOut(:sample, :inverse_prob) => Output(:inverse_prob),
            CompOut(:sample, :value) => Output(:value),
        ),
        c
    )