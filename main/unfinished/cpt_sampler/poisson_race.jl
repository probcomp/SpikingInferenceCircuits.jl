"""
    abstract type CatValCPTSampler <: ConcreteComponent end

A CPT sampler with outputs and inputs in the categorical value encoding.
"""
abstract type SpikingCatValCPTSampler <: GenericComponent end
target(::Type{<:SpikingCatValCPTSampler}) = Spiking()
inputs(c::SpikingCatValCPTSampler) = CompositeValue(
    Tuple(SpikingCategoricalValue(v) for v in values(output(abstract(c))))
)
outputs(c::SpikingCatValCPTSampler) = CompositeValue((SpikingCategoricalValue(output(abstract(c))[1]),))

struct PoissonRaceCPTSampler{C} <: CatValCPTSampler
    cpt::C
end
abstract(s::PoissonRaceCPTSampler) = AbstractCPTSampler(s.cpt)

struct PoissonRaceFixedDistSampler <: CatValCPTSampler
    rate::Float64
    dist::Categorical
end
component_graph(p::PoissonRaceFixedDistSampler) =
    ComponentGraph(1, ncategories(p.dist)) do (input,), outputs
        let neurons = [PoissonNeuron(rate * p) for p in p.dist]
            (neurons, Iterators.flatten((
                (input => input(neuron)[:on] for neuron in neurons),
                (output(neuron) => out for (neuron, out) in zip(neurons, outputs)),
                (output(n1) => input(n2)[:off] for n1 in neurons for n2 in neurons)
            )))
        end
    end