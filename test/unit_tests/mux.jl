og = PoissonAsyncOnGate(
    ConcreteAsyncOnGate(300, 0.1, 50),
    0., 2.
)

for num_in=0:5
    dict = simulate_get_output_evts(
        implement_deep(og, Spiking()),
        20,
        inputs=[
            (0., (:on,)),
            (
                (i, (:in,))
                for i=1:num_in
            )...
        ]
    ) |> spiketrain_dict
    @test length(dict) == (num_in > 0 ? 1 : 0)
    if num_in > 0
        @test haskey(dict, "nothing: out")
        @test length(dict["nothing: out"]) == num_in
    end 
end
