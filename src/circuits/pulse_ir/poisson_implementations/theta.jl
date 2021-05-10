struct PoissonTheta <: ConcretePulseIRPrimitive
    n_possibilities::Int
    M::Float64
    ΔT::Float64
    rate_rescaling_factor::Float64
    wta_off::PoissonOffGate
end

Circuits.abstract(θ::PoissonTheta) = Theta(θ.n_possibilities)
Circuits.target(θ::PoissonTheta) = target(abstract(θ))
Circuits.inputs(θ::PoissonTheta) = inputs(abstract(θ))
Circuits.outputs(θ::PoissonTheta) = outputs(abstract(θ))

Circuits.implement(θ::PoissonTheta, ::Spiking) =
    CompositeComponent(
        inputs(θ), outputs(θ),
        (
            neurons=IndexedComponentGroup(
                PoissonNeuron([c -> min(1, c) × θ.M, c -> c], θ.ΔT, u -> max(0., u - θ.M)/θ.rate_rescaling_factor)
                for _=1:θ.n_possibilities
            ),
            wta=ConcreteWTA(θ.n_possibilities, θ.wta_off)
        ),
        (
            (
                Input(:go) => CompIn(:neurons => i, 1)
                for i=1:θ.n_possibilities
            )...,
            (
                Input(:probs => i) => CompIn(:neurons => i, 2)
                for i=1:θ.n_possibilities
            )...,
            (
                CompOut(:neurons => i, :out) => CompIn(:wta, i)
                for i=1:θ.n_possibilities
            )...,
            (
                CompOut(:wta, i) => Output(i)
                for i=1:θ.n_possibilities
            )...
        ),
        θ
    )

# TODO: temporal interface
