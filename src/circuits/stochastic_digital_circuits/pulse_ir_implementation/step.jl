struct PulseStep <: GenericComponent
    input::Value
end
Circuits.abstract(s::PulseStep) = Step(s.input)
Circuits.target(::PulseStep) = Spiking()
Circuits.inputs(s::PulseStep) = inputs(abstract(s))
Circuits.outputs(s::PulseStep) = outputs(abstract(s))
Circuits.implement(s::Step, ::Spiking) = PulseStep(s.input)

is_cluster(::FiniteDomainValue) = true
is_cluster(::SpikeWire) = true
is_cluster(::Value) = false

implement_to_clusters(v::CompositeValue) = CompositeValue(map(implement_to_clusters, v.vals))
implement_to_clusters(v::Value) = is_cluster(v) ? v : implement_to_clusters(implement(v, Spiking()))

pairs_deep(val) = (k => val[k] for k in keys_deep(val))

cluster_sizes(val) = [
    implement_deep(val, Spiking()) |> 
            (x -> x isa SpikeWire ? 1 : x |> keys_deep |> collect |> length)
    for (_, val) in pairs_deep(implement_to_clusters(val))
]

Circuits.implement(s::PulseStep, ::Spiking) =
    CompositeComponent(
        implement_deep(inputs(s), Spiking()),
        implement_deep(outputs(s), Spiking()),
        (sync=PulseIR.Sync(cluster_sizes(s.input)),),
        (
            (
                if val isa SpikeWire
                    Input(:in => key) => CompIn(:sync, i => 1)
                else
                    Input(:in => key) => CompIn(:sync, i)
                end
                for (i, (key, val)) in enumerate(pairs_deep(implement_to_clusters(s.input)))
            )...,
            (
                if val isa SpikeWire
                    CompOut(:sync, i => 1) => Output(:out => key)
                else
                    CompOut(:sync, i) => Output(:out => key)
                end
                for (i, (key, val)) in enumerate(pairs_deep(implement_to_clusters(s.input)))
            )...
        ),
        s
    )