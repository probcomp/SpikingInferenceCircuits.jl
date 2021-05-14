struct PoissonTheta <: ConcretePulseIRPrimitive
    n_possibilities::Int
    M::Float64 # num spikes needed to override and have sample to early
    L::Float64 # P[θ samples index j | P[j]=0] ∝ eᴸ
    ΔT::Float64 # memory time
    rate::Float64 # rate at which samples are output
    wta_off::PoissonOffGate # used to construct the WTA to choose a sample
    function PoissonTheta(n, M, L, ΔT, rate, wta_off)
        if exp(L) > 0.01
            @warn "Setting $L is a pretty large value for L, and could significantly skew the distribution the θ gate samples from."
        end
        return new(n, M, L, ΔT, rate, wta_off)
    end
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
                # These equations were worked out on page 94-95 of notebook
                # Idea is that if ∑cⱼ = 0, each output is equally likely,
                # and if ∑cⱼ ≥ 1, the probability of sampling i is almost proportional
                # to cᵢ (though off by a small amount proportional to eᴸ).
                PoissonNeuron([
                        c -> min(1, c) × θ.M,
                        cᵢ -> cᵢ == 0 ? θ.L : log(cᵢ),
                        ∑cⱼ -> ∑cⱼ == 0 ? -log(θ.n_possibilities) - θ.L : -log(∑cⱼ)
                    ],
                    θ.ΔT,
                    u -> θ.rate × exp(u - θ.M)
                )
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
                Input(:probs => j) => CompIn(:neurons => j, 3)
                for i=1:θ.n_possibilities
                    for j=1:θ.n_possibilities
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



# struct PoissonTheta <: ConcretePulseIRPrimitive
#     n_possibilities::Int
#     M::Float64
#     ΔT::Float64
#     # if a sample got `C` spikes, the corresponding neuron will spike at rate C/rate_rescaling_factor
#     rate_rescaling_factor::Float64
#     wta_off::PoissonOffGate
# end

# Circuits.abstract(θ::PoissonTheta) = Theta(θ.n_possibilities)
# Circuits.target(θ::PoissonTheta) = target(abstract(θ))
# Circuits.inputs(θ::PoissonTheta) = inputs(abstract(θ))
# Circuits.outputs(θ::PoissonTheta) = outputs(abstract(θ))

# Circuits.implement(θ::PoissonTheta, ::Spiking) =
#     CompositeComponent(
#         inputs(θ), outputs(θ),
#         (
#             neurons=IndexedComponentGroup(
#                 PoissonNeuron([c -> min(1, c) × θ.M, c -> c], θ.ΔT, u -> max(0., u - θ.M)/θ.rate_rescaling_factor)
#                 for _=1:θ.n_possibilities
#             ),
#             wta=ConcreteWTA(θ.n_possibilities, θ.wta_off)
#         ),
#         (
#             (
#                 Input(:go) => CompIn(:neurons => i, 1)
#                 for i=1:θ.n_possibilities
#             )...,
#             (
#                 Input(:probs => i) => CompIn(:neurons => i, 2)
#                 for i=1:θ.n_possibilities
#             )...,
#             (
#                 CompOut(:neurons => i, :out) => CompIn(:wta, i)
#                 for i=1:θ.n_possibilities
#             )...,
#             (
#                 CompOut(:wta, i) => Output(i)
#                 for i=1:θ.n_possibilities
#             )...
#         ),
#         θ
#     )

# TODO: temporal interface
