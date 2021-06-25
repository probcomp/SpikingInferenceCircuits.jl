struct PoissonAsyncOnGate <: ConcretePulseIRPrimitive
    gate::ConcreteAsyncOnGate
    offrate::Float64
    onrate::Float64
end
Circuits.abstract(g::PoissonAsyncOnGate) = g.gate
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonAsyncOnGate) = Circuits.$s(Circuits.abstract(g)))
end

async_on_gate(off::PoissonOffGate) = PoissonAsyncOnGate(async_on_gate(abstract(off)), off.offrate, off.onrate)

Circuits.implement(g::PoissonAsyncOnGate, ::Spiking) =
    CompositeComponent(
        inputs(g), outputs(g),
        (neuron=PoissonNeuron(
            [
                x -> x,
                (let M = g.gate.M; x -> M*min(x, 1); end),
                x -> -x
            ], g.gate.ΔT,
            truncated_linear(g.offrate, g.onrate, g.gate.M, g.gate.M + 1)
        ),),
        (
            Input(:in) => CompIn(:neuron, 1),
            Input(:on) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        ),
        g
    )

# failure_probability_bound(g::PoissonAsyncOnGate) =
#     1 - (1 - p_spikes_while_off_bound(g))*(1 - p_doesnt_spike_by_delay_bound(g))

# p_spikes_while_off_bound(g::PoissonAsyncOnGate) =
#     1 - exp(-g.gate.ΔT × exp(-g.R/2))

# # Derived on page 76 of notebook
# p_doesnt_spike_by_delay_bound(g::PoissonAsyncOnGate) =
#     let α = g.gate.max_delay × (exp(g.R) - 1)
#         1 - (1 - exp(-α × exp(-g.R/2)))^(g.gate.M - 1)
#     end