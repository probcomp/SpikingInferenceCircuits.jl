using Distributions: probs

"""
    CPTSampleScore <: GenericComponent

Unit to sample from a CPT, or observe the output value, and output
the probability of the sample/observation.
"""
struct CPTSampleScore <: GenericComponent
    cpt::CPT
    sample::Bool
end

Circuits.inputs(c::CPTSampleScore) = NamedValues(
    :in_vals => IndexedValues(
        FiniteDomainValue(n)
        for n in input_ncategories(c.cpt)
    ),
    (c.sample ? () : (:obs => FiniteDomainValue(ncategories(cpt)),))...
)
Circuits.outputs(c::CPTSampleScore) =
    NamedValues(
        (c.sample ? (:value => FiniteDomainValue(ncategories(c.cpt)),) : ())...,
        :prob => PositiveReal()
    )

assmt_cond_prob_matrix(cpt::CPT) =
    let a = assmts(cpt)
        (collect ∘ transpose)(hcat([
            probs(cpt[a[i]])
            for i=1:length(a)
        ]...))
    end

# default implementation by using a `ToAssmts` to create a "variable"
# with a different value for each assignment of the parent variables,
# and a `ConditionalSampleScore` on this assignment variable
generic_implementation(c::CPTSampleScore) =
    CompositeComponent(
        inputs(c), outputs(c),
        (
            assmts=ToAssmts(input_ncategories(c.cpt)),
            sample_score=ConditionalSampleScore(
                assmt_cond_prob_matrix(c.cpt),
                c.sample,
            )
        ),
        Iterators.flatten((
            (
                Input(:in_vals => i) => CompIn(:assmts, i)
                for i=1:num_inputs(c.cpt)
            ),
            (CompOut(:assmts, :out) => CompIn(:sample_score, :in_val),),
            (CompOut(:sample_score, :prob) => Output(:prob),),
            (c.sample ? (CompOut(:sample_score, :value) => Output(:value),) : ()),
            (!c.sample ? (Input(:obs) => CompIn(:sample_score, :obs),) : ())
        )),
        c
    )