struct PoissonOffGate <: ConcretePulseIRPrimitive
    gate::ConcreteOffGate
    onrate::Float64
    offrate::Float64
end
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
            let on = g.onrate, off = g.offrate
                u -> truncated_linear(off, on, -1/2, 1/2)
            end
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