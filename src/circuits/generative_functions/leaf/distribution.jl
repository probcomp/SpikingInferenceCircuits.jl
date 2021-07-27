################
# Distribution #
################

struct DistributionGenFn{Op} <: GenFn{Op}
    is_observed::Bool
    cpt::CPT
    DistributionGenFn{Propose}(cpt::CPT) = new{Propose}(false, cpt)
    DistributionGenFn{Generate}(is_observed::Bool, cpt::CPT) = new{Generate}(is_observed, cpt)
end
DistributionGenFn(cpt::CPT, op::Propose) = DistributionGenFn{Propose}(cpt)
DistributionGenFn(cpt::CPT, op::Generate) = DistributionGenFn{Generate}(op.observed_addrs == AllSelection(), cpt)

input_domains(d::DistributionGenFn) = Tuple(FiniteDomain(n) for n in input_ncategories(d.cpt))
output_domain(d::DistributionGenFn) = FiniteDomain(ncategories(d.cpt))
has_traceable_value(d::DistributionGenFn) = true
traceable_value(d::DistributionGenFn) = to_value(output_domain(d))
operation(d::DistributionGenFn{Generate}) = Generate(d.is_observed ? AllSelection() : EmptySelection())
score_value(::Distribution) = SingleNonnegativeReal()
score_value(d::DistributionGenFn{Generate}) = # TODO: better handle non-assess Generate
    operation(d) == Assess() ? ProbEstimate() : SingleNonnegativeReal()
score_value(::DistributionGenFn{Propose}) = ReciprocalProbEstimate()

function Circuits.implement(d::DistributionGenFn, t::Target)
    println("Implementing a distribution...")
    impl = _implement(d, t)
    println("Finished implementing a distribution.")
    return impl
end

_implement(d::DistributionGenFn{Propose}, ::Target) =
    sample_distribution_implementation(d; output_inverse_prob=true)
_implement(d::DistributionGenFn{Generate}, ::Target) =  
    if d.is_observed
        score_distribution_implementation(d)
    else
        # TODO: I have not tested this yet
        sample_distribution_implementation(d; output_inverse_prob=false)
    end

sample_distribution_implementation(d; output_inverse_prob) =
    RelabeledIOComponent(
        CPTSample(d.cpt),
        (:in_vals => :inputs,),
        (
            :value => (:trace, :value),
            :inverse_prob => output_inverse_prob ? :score : nothing
        ); abstract=d
    )
score_distribution_implementation(d) =
    RelabeledIOComponent(
        CPTScore(d.cpt),
        (:in_vals => :inputs,), (:prob => :score,), (:obs => :value,); abstract=d
    )

gen_fn_circuit(g::CPT, _, op) = DistributionGenFn(g, op)
gen_fn_circuit(::Gen.Distribution, _, _) = error("To be compiled to a circuit, all distributions must be CPTs.")