ti = PoissonThresholdedIndicator(5, 50, 1., 50, 0., 2.)

dict = simulate_get_output_evts(
    implement_deep(ti, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:4
        )...
    ]
) |> spiketrain_dict

@test length(dict) == 0

dict = simulate_get_output_evts(
    implement_deep(ti, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:5
        )...
    ]
) |> spiketrain_dict

@test length(dict) == 1
@test length(dict["nothing: out"]) == 1

dict = simulate_get_output_evts(
    implement_deep(ti, Spiking()),
    20,
    inputs=[
        (
            (i, (:in,))
            for i=1:10
        )...
    ]
) |> spiketrain_dict

@test length(dict) == 1
@test length(dict["nothing: out"]) == 1