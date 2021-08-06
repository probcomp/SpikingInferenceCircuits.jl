struct PulseTheta{Th, Ti} <: ConcretePulseIRPrimitive
    input_denominator::Float64 # TODO: handle when there are different denominators
    θ::Th
    ti::Ti
    PulseTheta(i, th::Th, ti::Ti) where {Th, Ti} = new{Th, Ti}(i, th, ti)
end

PulseTheta(
    input_denominator,
    n_inputs,
    theta_type, M, L, ΔT, rate,
    offgate,
    ti_type, ti_params

) = PulseTheta(
    input_denominator,
    theta_type(n_inputs, M, L, ΔT, rate, offgate),
    ti_type(n_inputs, ti_params...)
)

n_possibilities(θ::PulseTheta) = length(outputs(θ.θ))
Circuits.abstract(θ::PulseTheta) = Theta(n_possibilities(θ))
Circuits.target(::PulseTheta) = Spiking()
Circuits.inputs(θ::PulseTheta) =
    IndexedValues(
        IndicatedSpikeCountReal(UnbiasedSpikeCountReal(θ.input_denominator))
        for _=1:n_possibilities(θ)
    )
Circuits.outputs(θ::PulseTheta) = NamedValues(
    :val => SpikingCategoricalValue(n_possibilities(θ))
)

Circuits.implement(θ::PulseTheta, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(θ), Spiking()),
        implement_deep(outputs(θ), Spiking()),
        (
            ti=θ.ti,
            θ=θ.θ
        ),
        (
            (
                Input(i => :count) => CompIn(:θ, :probs => i)
                for i=1:n_possibilities(θ)
            )...,
            (
                Input(i => :ind) => CompIn(:ti, :in)
                for i=1:n_possibilities(θ)
            )...,
            CompOut(:ti, :out) => CompIn(:θ, :go),
            (
                CompOut(:θ, i) => Output(:val => i)
                for i=1:n_possibilities(θ)
            )...
        ),
        θ
    )