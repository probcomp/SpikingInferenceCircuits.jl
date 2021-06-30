og = PoissonAsyncOnGate(
    ConcreteAsyncOnGate(300, 0.1, 50),
    0., 2.
)

for num_in=0:5
    events = simulate_get_output_evts(
        implement_deep(og, Spiking()),
        20,
        inputs=[
            (0., (:on,)),
            (
                (i, (:in,))
                for i=1:num_in
            )...
        ]
    )
    dict = spiketrain_dict(
        filter(events) do (t, compname, event)
            (compname === :ss || compname === nothing) && event isa Sim.OutputSpike
        end
    )
    @test length(dict) == (num_in > 0 ? 1 : 0)
    if num_in > 0
        @test haskey(dict, "nothing: out")
        @test length(dict["nothing: out"]) == num_in
    end 
end