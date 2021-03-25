struct SpikingCPTSampleScore <: GenericComponent
    abst::CPTSampleScore
    overall_no_input_rate::Float64
    overall_post_input_rate::Float64
end
Circuits.target(::SpikingCPTSampleScore) = Spiking()
Circuits.abstract(s::SpikingCPTSampleScore) = s.abst
Circuits.inputs(s::SpikingCPTSampleScore) =
    implement(inputs(abstract(s)), Spiking())

Circuits.outputs(s::SpikingCPTSampleScore) =
    map(outputs(abstract(s))) do v
        if v isa PositiveReal
            SpikeRateReal(s.overall_post_input_rate)
        else
            implement(v, Spiking())
        end
    end

Circuits.implement(s::SpikingCPTSampleScore, ::Spiking) =
    CompositeComponent(
        generic_implementation(abstract(s)),
        input=implement(inputs(s), Spiking()),
        output=outputs(s),
        subcomponent_map=(c -> (
            if c isa ConditionalSampleScore
                SpikingConditionalSampleScore(c, s.overall_no_input_rate, s.overall_post_input_rate)
            else
                c
            end
        )),
        abstract=s
    )