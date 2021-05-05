struct PoissonAsyncOnGate <: ConcretePulseIRPrimitive
    gate::ConcreteAsyncOnGate
    # `R` controls the rate during ON and OFF modes.
    # "off" rate will always be ≤ exp(-R/2); "on" rate will always be ≥ exp(R/2)
    R::Float64
end
Circuits.abstract(g::PoissonAsyncOnGate) = g.gate
for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonAsyncOnGate) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(g::PoissonAsyncOnGate, ::Spiking) =
    CompositeComponent(
        inputs(g), outputs(g),
        (neuron=PoissonNeuron([
            x -> x, x -> g.gate.M*min(x, 1), x -> -x
        ], g.gate.ΔT, u -> exp(g.R*(u - g.gate.M - 1/2))),),
        (
            Input(:in) => CompIn(:neuron, 1),
            Input(:on) => CompIn(:neuron, 2),
            CompOut(:neuron, :out) => CompIn(:neuron, 3),
            CompOut(:neuron, :out) => Output(:out)
        ),
        g
    )

failure_probability_bound(g::PoissonAsyncOnGate) =
    1 - (1 - p_spikes_while_off_bound(g))*(1 - p_doesnt_spike_by_delay_bound(g))

p_spikes_while_off_bound(g::PoissonAsyncOnGate) =
    1 - exp(-g.gate.ΔT × exp(-g.R/2))

# Derived on page 76 of notebook
p_doesnt_spike_by_delay_bound(g::PoissonAsyncOnGate) =
    let α = g.gate.max_delay × (exp(g.R) - 1)
        1 - (1 - exp(-α × exp(-g.R/2)))^(g.gate.M - 1)
    end