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
                        (let M = θ.M; (c -> min(1, c) × M); end),
                        (let L = θ.L; (cᵢ -> cᵢ == 0 ? L : log(cᵢ)); end),
                        (let N = θ.n_possibilities, L = θ.L
                            (∑cⱼ -> begin
                                # println("sum = $(∑cⱼ)")
                                ∑cⱼ == 0 ? -log(N) - L : -log(∑cⱼ)
                            end)
                        end)
                    ],
                    θ.ΔT,
                    let M = θ.M, rate = θ.rate
                        u -> begin
                            # println("u = $u; Rate = $(rate × exp(u - M))")
                            rate × exp(u - M)
                        end
                    end
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
                Input(:probs => j) => CompIn(:neurons => i, 3)
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