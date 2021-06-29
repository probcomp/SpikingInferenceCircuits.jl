ti = PoissonThresholdedIndicator(5, 50, 1., 50, 0., 2.)

events = Sim.simulate_for_time_and_get_events(
    implement_deep(ti, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:4
        )...
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

@test length(dict) == 0

events = Sim.simulate_for_time_and_get_events(
    implement_deep(ti, Spiking()),
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
@test length(dict["nothing: out"]) == 1

events = Sim.simulate_for_time_and_get_events(
    implement_deep(ti, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:10
        )...
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing && event isa Sim.OutputSpike
    end
)

@test length(dict) == 1
@test length(dict["nothing: out"]) == 1