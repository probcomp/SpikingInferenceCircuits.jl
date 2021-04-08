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
Circuits.implement(d::DistributionGenFn, ::Target) =
    genfn_from_cpt_sample_score(CPTSampleScore(d.cpt, true), d, !d.is_observed)

gen_fn_circuit(g::CPT, _, op) = DistributionGenFn(g, op)
gen_fn_circuit(::Gen.Distribution, _, _) = error("To be compiled to a circuit, all distributions must be CPTs.")