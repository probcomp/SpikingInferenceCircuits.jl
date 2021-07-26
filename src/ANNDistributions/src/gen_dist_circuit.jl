# Currently only supports Propose
struct ANNDistGenFn <: SIC.GenFn{Propose}
    input_ncategories
    out_ncategories
    ann
end
SIC.operation(::ANNDistGenFn) = Propose()
SIC.input_domains(d::ANNDistGenFn) = Tuple(SIC.FiniteDomain(n) for n in d.input_ncategories)
SIC.output_domain(d::ANNDistGenFn) = SIC.FiniteDomain(d.out_ncategories)
SIC.has_traceable_value(d::ANNDistGenFn) = true
SIC.traceable_value(d::ANNDistGenFn) = SIC.to_value(SIC.output_domain(d))
SIC.score_value(::ANNDistGenFn) = SIC.ReciprocalProbEstimate()

params() = (100., 1000., timer_params())
Circuits.implement(d::ANNDistGenFn, ::Spiking) =
    Circuits.RelabeledIOComponent(
        ANNCPTSample(d.ann, d.input_ncategories),
        (:in_vals => :inputs,),
        (
            :value => (:trace, :value),
            :inverse_prob => :score
        ),
        abstract=d
    )

SIC.gen_fn_circuit(a::ANN_LCPT, arg_domains, ::Propose) =
    ANNDistGenFn(map(length, a.in_domains), length(a.out_labels), a.ann)