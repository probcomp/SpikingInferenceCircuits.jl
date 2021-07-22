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

# Hardcode some parameters for compilation
# TODO: a less ad-hoc mechanism for setting these!
module Params
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
include("../../../experiments/utils/default_implementation_rules.jl")
timer_params() = (
    NSPIKES_SYNC_TIMER(),  #N_spikes_timer
    (1, M(), GATE_RATES()...), 0., # timer TI params (maxdelay M gaterates...) | offrate
    neuron_ΔT() * 10 # TODO: this should depend on n_layers!!!
)
params() = (100., 1000., timer_params())
end

Circuits.implement(d::ANNDistGenFn, ::Spiking) =
    Circuits.RelabeledIOComponent(
        ANNCPTSample(d.ann, Params.params()..., d.input_ncategories),
        (:in_vals => :inputs,),
        (
            :value => (:trace, :value),
            :inverse_prob => :score
        ),
        abstract=d
    )

SIC.gen_fn_circuit(a::ANN_LCPT, arg_domains, ::Propose) =
    ANNDistGenFn(map(length, a.in_domains), length(a.out_labels), a.ann)