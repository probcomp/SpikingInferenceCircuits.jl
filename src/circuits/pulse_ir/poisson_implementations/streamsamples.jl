struct PoissonStreamSamples <: ConcretePulseIRPrimitive
    P::Matrix{Float64}
    overall_on_rate::Float64
    overall_off_rate::Float64
    ΔT::Float64 # Neuron memory (time before neurons turn off)
end
Circuits.abstract(p::PoissonStreamSamples) = StreamSamples(p.P, p.ΔT,
    t -> Distributions.Poisson(t * p.overall_on_rate)
)

for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonStreamSamples) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(p::PoissonStreamSamples, ::Spiking) =
    let bias = log(p.overall_off_rate),
        base_weight = log(c.overall_on_rate) - bias
            CompositeComponent(
                inputs(p), outputs(p),
                Tuple(
                    PoissonNeuron(
                        [
                            x -> min(1, x) × (
                                prob_output_given_input(
                                    abstract(c), outval
                                ) .+ base_weight
                            )
                        ],
                        ΔT, u -> exp(u - bias)
                    )
                    for outval = 1:out_domain_size(abstract(p))
                ),
                Iterators.flatten((
                    (
                        Input(inval) => CompIn(outval, 1)
                        for outval=1:out_domain_size(abstract(p))
                            for inval=1:in_domain_size(abstract(p))
                    ),
                    (
                        CompOut(outval, :out) => Output(outval)
                        for outval=1:out_domain_size(abstract(p))
                    )
                ))
            )
    end

### Temporal Interface ###

# The only failure mode is emitting a spike before turned on.
failure_probabability_bound(p::PoissonStreamSamples) = exp(-p.overall_off_rate)

# can_support_inwindows(p::PoissonStreamSamples, d::Dict{Input, Window}) =

valid_strict_inwindows(::PoissonStreamSamples, ::Dict{Input, Window}) = error("Not implemented.")

output_windows(ss::PoissonStreamSamples, d::Dict{Input, Window}) =
    let inw = d[Input(:in)],
        outwindow = Window(
        Interval(inw.interval.min, inw.interval.max + ss.ΔT),
        inw.pre_hold, Inf
    )
        Dict(Output(i) => outwindow for i=1:out_domain_size(abstract(ss)))
    end