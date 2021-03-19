"""
    SpikingConditionalSampleScore

Implementation of `ConditionalSampleScore` for `Spiking`, using `IntegratingPoisson`.
"""
struct SpikingConditionalSampleScore <: GenericComponent
    abst::ConditionalSampleScore
    overall_no_input_rate::Float64
    overall_post_input_rate::Float64
end
Circuits.abstract(c::SpikingConditionalSampleScore) = c.abst
Circuits.target(::SpikingConditionalSampleScore) = Spiking()

Circuits.inputs(c::SpikingConditionalSampleScore) =
    implement(inputs(abstract(c)), Spiking())
Circuits.outputs(c::SpikingConditionalSampleScore) =
    NamedValues((key => (
        if key == :prob
            SpikeRateReal(c.overall_post_input_rate)
        else
            implement(val, Spiking())
        end
        ) for (key, val) in pairs(outputs(abstract(c)))
    )...)

Circuits.implement(c::SpikingConditionalSampleScore, ::Spiking) =
    let bias = log(c.overall_no_input_rate)
        base_weight = log(c.overall_post_input_rate) - bias
            CompositeComponent(
                implement_deep(inputs(c), Spiking()),
                implement_deep(outputs(c), Spiking()),
                (;(
                    :spikers => IndexedComponentGroup(
                        IntegratingPoisson(
                            log.(prob_output_given_input(abstract(c), outval)) .+ base_weight,
                            bias,
                            exp
                        )
                        for outval=1:out_domain_size(abstract(c))
                    ),
                    :mux => Mux(out_domain_size(abstract(c)), NamedValues(:val => SpikeWire())),

                    # include buffer if we need to sample--
                    (abstract(c).sample ? (:cvb => CVB(out_domain_size(abstract(c))),) : ())...,
                )...),
                Iterators.flatten((
                    Iterators.flatten((
                        (Input(:in_val => inval) => CompIn(:spikers => outval, inval) for inval=1:in_domain_size(abstract(c)))...,
                        (CompOut(:spikers => outval, :out) => CompIn(:mux, :values => outval => :val)),
                        (abstract(c).sample ? sampler_outval_connections(outval) : obs_outval_connections(outval) )...
                    ) for outval=1:out_domain_size(abstract(c))),
                    (CompOut(:mux, :val) => Output(:prob),)
                ))
            )
    end

sampler_outval_connections(outval) = (
    (CompOut(:spikers => outval, :out) => CompIn(:cvb, outval)),
    (CompOut(:cvb, outval) => CompIn(:mux, :sel => outval)),
    (CompOut(:cvb, outval) => Output(:sample => outval))
)
obs_outval_connections(outval) = (
    (Input(:obs => outval) => CompIn(:mux, :sel => outval)),
)