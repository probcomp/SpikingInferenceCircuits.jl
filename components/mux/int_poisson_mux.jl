"""
    IPoissonBitMux <: GenericComponent

Implementation of `BitMux` for `Spiking` using an `IntegratingPoisson`.
Once an input is selected, it is selected forever.
"""
struct IPoissonBitMux <: GenericComponent end
Circuits.inputs(::BitMux) = NamedValues(:value => SpikeWire(), :sel => SpikeWire())
Circuits.outputs(::BitMux) = NamedValues(:out => SpikeWire())
# TODO: Don't hardcode the rates here!
Circuits.implement(m::BitMux, ::Spiking) = CompositeComponent(
        inputs(m), outputs(m),
        (
            inhibitor=IPoissonGatedRepeater(),
            outputter=IntegratingPoisson([+10., -10.], 0., ReLU)
        ),
        (
            Input(:sel) => CompIn(:inhibitor, :off),
            Input(:value) => CompIn(:inhibitor, :in),
            Input(:value) => CompIn(:outputter, 1),
            CompOut(:inhibitor, :out) => CompIn(:outputter, 2),
            CompOut(:outputter, :out) => CompIn(:outputter, 2),
            CompOut(:outputter, :out) => Output(:out)
        ),
        m
    )

ReLU(x) = max(x, 0.)
# struct IPoissonMux <: GenericComponent
#     mux::Mux
# end
# Circuits.inputs(ipm::IPoissonMux) = NamedValues(
#     :values => inputs(ipm.mux)[:values],
#     :sel => implement(inputs(ipm.mux)[:sel], Spiking())
# )
# Circuits.outputs(ipm::IPoissonMux) = outputs(ipm.mux)

# Circuits.abstract(ipm::IPoissonMux) = ipm.mux

# Circuits.implement(ipm::IPoissonMux, ::Spiking) =
#     let all_inputs = implement_deep(inputs(ipm)),
#         all_ouptuts = implement_deep(outputs(ipm)),
#     CompositeComponent(
#         all_inputs, all_outputs,
#         IPoissonGatedRepeater()
#     )