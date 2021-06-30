offgate = PoissonOffGate(
    ConcreteOffGate(300, 1., 50),
    0., 2.
)

events = simulate_get_output_evts(
    implement_deep(offgate, Spiking()),
    20,
    inputs=[
        (0., (:off,)),
        (
            (i, (:in,))
            for i=1:5
        )...
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

display(dict)

@test length(dict) == 0

events = simulate_get_output_evts(
    implement_deep(offgate, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:5
        )...
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

display(dict)

@test length(dict) == 1
@test length(dict["nothing: out"]) == 5

dict = simulate_get_output_evts(
    implement_deep(offgate, Spiking()),
    20,
    inputs=[
        (
            (2*i, (:in,))
            for i=1:2
        )...,
        (6., (:off,)),
        (
            (2*i, (:in,))
            for i=4:5
        )...
    ]
) |> spiketrain_dict

@test length(dict) == 1
@test length(dict["nothing: out"]) == 2

