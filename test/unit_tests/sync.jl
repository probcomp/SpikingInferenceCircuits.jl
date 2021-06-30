Circuits.implement(s::PulseIR.Sync, ::Spiking) =
    PulseIR.PoissonSync(
        s.cluster_sizes,
        (1000, 0., 5.),
        (0.1, 0., 5.),
        (50., 4, (0.1, 1000, 0., 5.), 0., 100.)
    )

### Simple single use ### 

sync = PulseIR.Sync([2, 1, 1])
impl = implement_deep(sync, Spiking())
println("Implemented.")

get_events(impl) = simulate_get_output_evts(impl, 500.;
    inputs=[(0., (1 => 1, 2 => 1, 3 => 1))]
)
events = get_events(impl)

dict = out_st_dict(events)
display(dict)
@test length(dict) == 3
for v in values(dict)
    @test length(v) == 1
end

### Recurrent use with delay ###

sync = PulseIR.Sync([1, 2])
timer = PulseIR.PoissonTimer(200., 50, (0.1, 1000, 0, 5.), 0., 300.)

# idea is: pass 
comp = CompositeComponent(
    NamedValues(:go => SpikeWire()), NamedValues(:sync_out1 => SpikeWire(), :sync_out2 => SpikeWire(), :unsync_out1 => SpikeWire(), :unsync_out2 => SpikeWire()),
    (
        sync = sync,
        t11 = timer,
        t12 = timer,
        t13 = timer,
        t21 = timer,
        t22 = timer,
        t23 = timer
    ),
    (
        Input(:go) => CompIn(:t11, :start),
        CompOut(:t11, :out) => CompIn(:t12, :start),
        CompOut(:t12, :out) => CompIn(:t13, :start),
        CompOut(:t13, :out) => Output(:unsync_out1),

        Input(:go) => CompIn(:t21, :start),
        CompOut(:t21, :out) => CompIn(:t22, :start),
        CompOut(:t22, :out) => CompIn(:t23, :start),
        CompOut(:t23, :out) => Output(:unsync_out2),
        
        
        CompOut(:t13, :out) => CompIn(:sync, 1 => 1),
        CompOut(:t23, :out) => CompIn(:sync, 2 => 1),

        CompOut(:sync, 1 => 1) => CompIn(:t11, :start),
        CompOut(:sync, 2 => 1) => CompIn(:t21, :start),
        CompOut(:sync, 1 => 1) => Output(:sync_out1),
        CompOut(:sync, 2 => 1) => Output(:sync_out2)
    )
)

impl = implement_deep(comp, Spiking())
println("Implemented.")
get_events(impl) = simulate_get_output_evts(impl, 5000.; inputs=[(0., (:go,),)])

events = get_events(impl)

dict = out_st_dict(events)
@test length(dict) == 4
while !isempty(dict["nothing: sync_out1"])
    u1 = pop!(dict["nothing: unsync_out1"])
    u2 = pop!(dict["nothing: unsync_out2"])
    s1 = pop!(dict["nothing: sync_out1"])
    s2 = pop!(dict["nothing: sync_out2"])

    @test abs(s1 - s2) < 4
    @test u1 < s1 && u2 < s1
    @test abs(u1 - s1) < 4 || abs(u2 - s1) < 4
end