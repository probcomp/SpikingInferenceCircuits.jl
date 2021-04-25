"""
    genfn_from_cpt_sample_score(cpt_ss::CPTSampleScore, g::GenFn, val_to_trace::Bool)

A `CompositeComponent` which implements `g` using the given `cpt_ss`.  If `val_to_trace` is true,
the value output from `cpt_ss` is both output as `:value` and `:trace`; if this is false, it is only
output as `:value`.
"""
genfn_from_cpt_sample_score(cpt_ss, g, val_to_trace) = CompositeComponent(
        inputs(g), outputs(g),
        (cpt_sample_score=cpt_ss,),
        Iterators.flatten((
            (Input(:inputs => i) => CompIn(:cpt_sample_score, :in_vals => i) for i=1:length(inputs(g)[:inputs])),
            (
                CompOut(:cpt_sample_score, :value) => Output(:value),
                (has_prob_output(g) ? (CompOut(:cpt_sample_score, :prob) => Output(:prob),) : ())...,
                (val_to_trace ? (CompOut(:cpt_sample_score, :value) => Output(:trace),) : ())...
            )
        )),
        g
    )
