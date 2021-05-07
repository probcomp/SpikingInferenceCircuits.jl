using Circuits
includet("../pulse_ir.jl")
using .PulseIR: Interval, Window

println("OFFGATE tests:")

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

offgate = PulseIR.ConcreteOffGate(10, 0.2, 10)

valid_strict_inwindows = PulseIR.valid_strict_inwindows(offgate, inwindows)
println("valid_strict_inwindows: $valid_strict_inwindows")

outwindows = PulseIR.output_windows(offgate, inwindows)
println("outwindows: $outwindows")

implemented_offgate = PulseIR.PoissonOffGate(offgate, 16)
pF = PulseIR.failure_probability_bound(implemented_offgate)
println("Failure probability: $pF")

###
println()
println("CondScore tests:")

cs = ConditionalScore(
    [0.5 0.5; 0.5 0.5]
)

# Q: how to switch between methods for implementing a component?