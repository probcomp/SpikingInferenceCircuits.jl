"""
    PoissonSpikeCountMultiplier(
        conc::ConcreteSpikeCountMultiplier,
        ((timer_M::Float64, timer_R::Float64), timer_offrate::Float64),
        (offgate_M::Float64, offgate_R::Float64)
    )
"""
struct PoissonSpikeCountMultiplier <: ConcretePulseIRPrimitive
    conc::ConcreteSpikeCountMultiplier
    timer_params::Tuple{Tuple{Float64, Float64, Float64}, Float64}
    offgate_params::Tuple{Int, Float64, Float64} # M, offrate, onrate [we set ΔT and max_delay using `conc`]
    function PoissonSpikeCountMultiplier(conc::ConcreteSpikeCountMultiplier, args...)
        @assert conc.spikecount_dist isa PoissonOnErlangTime
        new(conc, args...)
    end
end

PoissonSpikeCountMultiplier(
    input_count_denominators::Tuple{Vararg{<:Real}},
    output_denominator::Real,
    erlang_shape::Real, # number spikes used by timer
    expected_output_time::Real,
    max_input_memory::Real,
    max_delay::Real,
    timer_params::Tuple{<:Tuple{<:Real, <:Real, <:Real}, <:Real},
    offgate_params::Tuple{<:Real, <:Real, <:Real}
) = PoissonSpikeCountMultiplier(
    ConcreteSpikeCountMultiplier(
        input_count_denominators,
        PoissonOnErlangTime(output_denominator, erlang_shape, expected_output_time),
        max_input_memory, max_delay
    ),
    timer_params, offgate_params
)

Circuits.target(::PoissonSpikeCountMultiplier) = Spiking()
Circuits.abstract(m::PoissonSpikeCountMultiplier) = m.conc
Circuits.inputs(m::PoissonSpikeCountMultiplier) = inputs(abstract(m))
Circuits.outputs(m::PoissonSpikeCountMultiplier) = outputs(abstract(m))

Circuits.implement(m::PoissonSpikeCountMultiplier, ::Spiking) =
    CompositeComponent(
        inputs(m), outputs(m),
        (
            timer=PoissonTimer(
                m.conc.spikecount_dist.expected_output_length,
                m.conc.spikecount_dist.erlang_shape,
                (m.conc.max_delay, m.timer_params[1]...), # Timer TI parameters
                m.timer_params[2], # Timer off rate
                m.conc.max_input_memory
            ),
            mult=PoissonNeuron(
                [
                    c -> log(c/denominator)
                    for denominator in m.conc.input_count_denominators
                ],
                m.conc.max_input_memory,
                let ratio = m.conc.spikecount_dist.K/m.conc.spikecount_dist.expected_output_length
                    u -> ratio × exp(u)
                end
            ),
            gate=PoissonOffGate(
                ConcreteOffGate(
                    m.conc.max_input_memory,
                    m.conc.max_delay,
                    m.offgate_params[1]
                ),
                m.offgate_params[2:end]...
            )
        ),
        (
            Input(:ind) => CompIn(:timer, :start),
            (
                Input(:counts => i) => CompIn(:mult, i)
                for i=1:length(inputs(m)[:counts])
            )...,
            CompOut(:mult, :out) => CompIn(:gate, :in),
            CompOut(:timer, :out) => CompIn(:gate, :off),
            CompOut(:timer, :out) => Output(:ind),
            CompOut(:gate, :out) => Output(:count)
        ),
        m
    )

# TODO: failure probability

