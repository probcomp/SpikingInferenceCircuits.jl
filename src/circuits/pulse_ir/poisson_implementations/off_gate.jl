struct PoissonOffGate <: ConcretePulseIRPrimitive
    gate::ConcreteOffGate
    offrate::Float64
    onrate::Float64
end
PoissonOffGate(ΔT::Real, max_delay::Real, M::Real, offrate::Real, onrate::Real) =
    PoissonOffGate(ConcreteOffGate(ΔT, max_delay, M), offrate, onrate)

Circuits.abstract(g::PoissonOffGate) = g.gate
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonOffGate) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(g::PoissonOffGate, ::Spiking) =
    CompositeComponent(
        inputs(g), outputs(g),
        (neuron=PoissonNeuron(
            [
                x -> x,
                let M = g.gate.M; x -> -M*x; end,
                x -> -x
            ], g.gate.ΔT,
        truncated_linear(g.offrate, g.onrate, 0, 1)
            # u -> begin
            #     println("u = $u, λ = $(truncated_linear(g.offrate, g.onrate, 0, 1)(u))")
            #     truncated_linear(g.offrate, g.onrate, 0, 1)
            # end
        ),),
        (
            Input(:in) => CompIn(:neuron, 1),
            Input(:off) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        ),
        g
    )

# failure_probability_bound(g::PoissonOffGate) =
#     1 - (1 - p_spikes_while_off_bound(g))*(1 - p_doesnt_spike_by_delay_bound(g))

# p_spikes_while_off_bound(g::PoissonOffGate) =
#     1 - exp(-g.gate.ΔT × exp(-g.R/2))

# # Derived on page 76 of notebook
# p_doesnt_spike_by_delay_bound(g::PoissonOffGate) =
#     let α = g.gate.max_delay × (exp(g.R) - 1)
#         1 - (1 - exp(-α × exp(-g.R/2)))^(g.gate.M - 1)
#     end