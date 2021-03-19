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
    implement(outputs(abstract(c)), Spiking())

Circuits.implement(c::SpikingConditionalSampleScore, ::Spiking) =
    let bias = log(c.overall_no_input_rate)
        base_weight = log(c.overall_post_input_rate) - bias
            CompositeComponent(
                implement_deep(inputs(c), Spiking()),
                implement_deep(outputs(c), Spiking()),
                (;(
                    :spikers => IndexedComponentGroup(
                        IntegratingPoisson(
                            abstract(c).P[:,y] .+ base_weight,
                            bias,
                            exp
                        )
                        for y=1:ysize(abstract(c))
                    ),
                    :mux => Mux(xsize(abstract(c)), NamedValues(:val => SpikeWire())),

                    # include buffer if we need to sample--
                    (abstract(c).sample ? (:cvb => CVB(xsize(abstract(c))),) : ())...,
                )...),
                Iterators.flatten((
                    Iterators.flatten((
                        (Input(:in_val => y) => CompIn(:spikers => x, y) for y=1:ysize(abstract(c)))...,
                        (CompOut(:spikers => x, :out) => CompIn(:mux, :values => x => :val)),
                        (abstract(c).sample ? sampler_x_connections(x) : obs_x_connections(x) )...
                    ) for x=1:xsize(abstract(c))),
                    (CompOut(:mux, :val) => Output(:prob),)
                ))
            )
    end

sampler_x_connections(x) = (
    (CompOut(:spikers => x, :out) => CompIn(:cvb, x)),
    (CompOut(:cvb, x) => CompIn(:mux, :sel => x)),
    (CompOut(:cvb, x) => Output(:sample => x))
)
obs_x_connections(x) = (
    (Input(:obs => x) => CompIn(:mux, :sel => x)),
)