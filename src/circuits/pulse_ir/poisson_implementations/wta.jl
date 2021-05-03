# TODO: get rid of this file?

struct PoissonWTA <: GenericComponent
    n_inputs::Int
end
Circuits.abstract(w::PoissonWTA) = WTA(w.n_inputs)

for s in (:target, :inputs, :outputs)
    @eval (Circuits.$s(g::PoissonStreamSamples) = Circuits.$s(Circuits.abstract(g)))
end

Circuits.implement(p::PoissonWTA, ::Spiking) =
    CompositeComponent(
        inputs(p), outputs(p),
        (
            spikers=
        )
    )