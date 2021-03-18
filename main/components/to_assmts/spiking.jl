struct SpikingToAssmts{n} <: GenericComponent
    size::NTuple{n, Int}
end
Circuits.abstract(s::SpikingToAssmts) = ToAssmts(s.size)
Circuits.target(::SpikingToAssmts) = Spiking()
Circuits.inputs(s::SpikingToAssmts) = implement(inputs(abstract(s), Spiking()))
Circuits.outputs(s::SpikingToAssmts) = implement(outputs(abstract(s), Spiking()))
num_inputs(::SpikingToAssmts{n}) = n

Circuits.implement(s::SpikingToAssmts) =
    CompositeComponent(
        Tuple(SpAND(num_inputs(s)) for _=1:length(outputs(s))),
        Iterators.flatten(
            
            for (assmt, outval) in pairs(LinearIndices(s.size))
        )
    )