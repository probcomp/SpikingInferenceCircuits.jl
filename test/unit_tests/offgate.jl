offgate = PoissonOffGate(
    ConcreteOffGate(300, 1., 50),
    0., 2.
)

events = Sim.simulate_for_time_and_get_events(
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

events = Sim.simulate_for_time_and_get_events(
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

@test length(dict) == 1
@test length(dict["nothing: out"]) == 5

events = Sim.simulate_for_time_and_get_events(
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
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

@test length(dict) == 1
@test length(dict["nothing: out"]) == 2

