using Circuits
includet("../pulse_ir.jl")
using .PulseIR: Interval, Window

println("OFFGATE tests:")
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

valid_strict_inwindows = PulseIR.valid_strict_inwindows(offgate, inwindows)
println("valid_strict_inwindows: $valid_strict_inwindows")

outwindows = PulseIR.output_windows(offgate, inwindows)
println("outwindows: $outwindows")

###
println()
println("CondScore tests:")

cs = ConditionalScore(
    [0.5 0.5; 0.5 0.5]
)
pcs = PulseConditionalScore(
    PoissonStreamSamples(cs.P, 1.0, 1/1000000, 20),
    implement(Mux, ), # TODO
    PoissonThresholdedIndicator(10, ),

)

# Q: how to switch between methods for implementing a component?