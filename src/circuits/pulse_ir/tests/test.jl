using Circuits
includet("../pulse_ir.jl")
using .PulseIR: Interval, Window

offgate = PulseIR.PoissonOffGate(10, 20, 10, 0.2)
pF = PulseIR.failure_probability_bound(offgate)
println("Failure probability: $pF")

inwindows = Dict{Input, Window}(
    Input(:off) => Window(
        Interval(0., 5.),
        0., 9.0
    ),
    Input(:in) => Window(
        Interval(5., 10.),
        2.0, 0.0
    )
)

supports_windows = PulseIR.can_support_inwindows(offgate, inwindows)
println("supports_windows: $supports_windows")

outwindows = PulseIR.output_windows(offgate, inwindows)
println("outwindows: $outwindows")