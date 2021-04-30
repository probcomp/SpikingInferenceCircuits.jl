"""
    WTA(n)

Winner takes all circuit with `n` input values.  Repeats the first spike passed in.
"""
struct WTA <: GenericComponent
    n_inputs::Int
end
Circuits.target(::WTA) = Spiking()
Circuits.inputs(w::WTA) = IndexedValues(SpikeWire() for _=1:w.n_inputs)
Circuits.outputs(w::WTA) = Circuits.inputs(w)
# TODO: `fail` wire output for when it fails to distinguish the first input