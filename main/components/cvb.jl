struct CVB <: GenericComponent
    n::Int
end
Circuits.target(::CVB) = Spiking()
Circuits.inputs(cvb::CVB) = IndexedValues(SpikeWire() for _=1:cvb.n)
Circuits.outputs(cvb::CVB) = inputs(cvb)

# TODO: have a single neuron that spikes to turn things off to decrease the number of edges?
Circuits.implement(cvb::CVB, ::Spiking) = CompositeComponent(
        inputs(cvb), outputs(cvb),
        Tuple(IPoissonGatedRepeater() for _=1:cvb.n),
        Iterators.flatten((
            (Input(i) => CompIn(i, :in) for i=1:cvb.n),
            (CompOut(i, :out) => Outputs(i, :out) for i=1:cvb.n),
            (CompOut(i, :out) => CompIn(j, :off) for i=1:cvb.n, j=1:cvb.n)
        ))
    )