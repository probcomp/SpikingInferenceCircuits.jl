"""
    SpikeCountMultiplier((K₁, …, Kₙ), K)

Circuit to multiply or approximately multiply `n` spike counts.
Given counts `c₁, …, cₙ`, outputs `∏cᵢ × K / ∏Kᵢ` spikes in expectation.

A spike must arrive in the `:ind` input when all the counts have finished being delivered;
after the correct output spike count has been emitted, a spike will occur in the `:ind` output line.
"""
struct SpikeCountMultiplier{n} <: GenericComponent
    # count_inputs::NTuple{n, UnbiasedSpikeCountReal}
    input_count_denominators::NTuple{n, Float64}
    output_count_denominator::Float64
end
Circuits.target(::SpikeCountMultiplier) = Spiking()
Circuits.inputs(m::SpikeCountMultiplier) = NamedValues(
    :counts => IndexedValues(SpikeWire() for _ in m.input_count_denominators),
    :ind => SpikeWire()
)
Circuits.outputs(::SpikeCountMultiplier) = NamedValues(
    :count => SpikeWire(),
    :ind => SpikeWire()
)

### Distributions for unbiased multiplication ###
"""
A type representing a distribution on the number of output counts
from a multiplier or an approximate multiplier, given the input counts.
"""
abstract type MultiplicationCountDistribution end

"""
The denominator for the `UnbiasedSpikeCountReal` output under
a given distribution.
""" # TODO: should we let the denominator depend on more than the distribution?
# E.g. also on the incoming counts?
output_count_denominator(::MultiplicationCountDistribution) =
    error("Not implemented.")

"""
Given input counts `c₁, …, cₙ` with denominators `K₁, …, Kₙ`,
this is the distribution on `C` from the sampling process
```
T ~ Erlang(rate=erlang_shape/expected_output_length, shape=erlang_shape)
C ~ Poisson(T × c/expected_output_length)
```
where the expected count `c = K × ∏cᵢ / ∏Kᵢ`.
"""
struct PoissonOnErlangTime <: MultiplicationCountDistribution
    K::Float64
    erlang_shape::Float64 # = number spikes we wait for from the timer
    expected_output_length::Float64
end
output_count_denominator(p::PoissonOnErlangTime) = p.K

### Concrete Spike Multiplier ###
struct ConcreteSpikeCountMultiplier{n, Dist} <: ConcretePulseIRPrimitive
    input_count_denominators::NTuple{n, Float64}
    spikecount_dist::Dist
    # Temporal interface parameters
    max_input_memory::Float64
    max_delay::Float64 # maximum delay between when memory ends and a spike from memory period manages to be output
    ConcreteSpikeCountMultiplier(
        inputs::NTuple{n, <:Real},
        dist::D, args...
    ) where {n, D <: MultiplicationCountDistribution} =
        new{n, D}(inputs, dist, args...)

end
Circuits.target(::ConcreteSpikeCountMultiplier) = Spiking()
Circuits.abstract(m::ConcreteSpikeCountMultiplier) =
    SpikeCountMultiplier(m.input_count_denominators, output_count_denominator(m.spikecount_dist))
Circuits.inputs(m::ConcreteSpikeCountMultiplier) = inputs(abstract(m))
Circuits.outputs(m::ConcreteSpikeCountMultiplier) = outputs(abstract(m))

### Temporal Interface ###
function output_windows(
    ::ConcreteSpikeCountMultiplier,
    d::Dict{Input, Window}
)
    # any valid strict joint input window ends when the indicator spike arrives
    # input_endtime = d[Input(:ind)].interval.max
    # count_window = containing_window(
    #     window for (key, window) in d if key != Input(:ind)
    # )

    # required_input_memory = (input_endtime - count_window.interval.min)
    # outwindow_size = d.max_input_memory - required_input_memory
    out_endtime = d[Input(:ind)].interval.min + d.max_input_memory + d.max_delay
    outwindow = Window(
            Interval(
                d[Input(:ind)].interval.min,
                out_endtime
            ),
            d[Input(:ind)].pre_hold,
            interval_length(intersect(
                post_hold_interval(d[Input(:ind)]),
                Interval(out_endtime, Inf)
            ))
        )
    return Dict{Output, Window}(
        Output(:count) => outwindow,
        Output(:ind) => outwindow
    )
end