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

# Question: should this be viewed as a primitive or not?
# On one hand, we can implement this using `OffGate`.
# on the other, I think we will need to describe its PulseIR
# temporal interface separately.
Circuits.implement(w::WTA) =
    CompositeComponent(
        inputs(w), outputs(w),
        Tuple(
            OffGate() for _=1:w.n_inputs
        ),
        Iterators.flatten((
            (
                Input(i) => CompIn(i, :in)
                for i=1:w.n_inputs
            ),
            (
                CompOut(i, :out) => CompIn(j, :off)
                for i=1:w.n_inputs
                    for j=1:w.n_inputs
            ),
            (
                CompOut(i, :out) => Output(i)
                for i=1:w.n_inputs
            )
        ))
    )

### Temporal Interface ###
