struct Resample <: GenericComponent
    n_possibilities::Int
    trace_value::CompositeValue
end
Circuits.inputs(r::Resample) = NamedValues(
    :weights => IndexedValues(
        NonnegativeReal() for _=1:r.n_possibilities
    ),
    :traces => IndexedValues(
        r.trace_value for _=1:r.n_possibilities
    )
)
Circuits.outputs(r::Resample) = NamedValues(
    :traces => IndexedValues(
        r.trace_value for _=1:r.n_possibilities
    )
)

Circuits.implement(r::Resample, ::Target) =
    CompositeComponent(
        inputs(r), outputs(r),
        (
            thetas=IndexedComponentGroup(
                Theta(r.n_possibilities) for _=1:r.n_possibilities
            ),
            muxes=IndexedComponentGroup(
                Mux(r.n_possibilities, r.trace_value)
                for _=1:r.n_possibilities
            )
        ),
        (
            (
                Input(:weights => i) => CompIn(:thetas => j, i)
                for i=1:r.n_possibilities
                    for j=1:r.n_possibilities
            )...,
            (
                CompOut(:thetas => i, :val) => CompIn(:muxes => i, :sel)
                for i=1:r.n_possibilities
            )...,
            (
                Input(:traces => i) => CompIn(:muxes => j, :values => i)
                for i=1:r.n_possibilities
                    for j=1:r.n_possibilities
            )...,
            (
                CompOut(:muxes => i, :out) => Output(:traces => i)
                for i=1:r.n_possibilities
            )...
        ),
        r
    )