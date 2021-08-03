using Test
using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
const Sim = SpikingCircuits.SpikingSimulator
import Distributions
using Distributions: Exponential

using .PulseIR: ConcreteStreamSamples, ConcreteThresholdedIndicator, ConcreteOffGate, ConcreteAsyncOnGate
using .PulseIR: PoissonStreamSamples, PoissonThresholdedIndicator, PoissonOffGate, PoissonAsyncOnGate
using .SDCs: PulseMux

includet("spiketrain_utils.jl")

N_Products = 3
Terms_Per_Product = 3
Term_Denominator = 20
Renorm_Rate = 0.1
base = 2
TI_Threshold = 5
Memory = 200 # ms

# rate at which counts are fed into unit
Count_Inrate = 0.4
MaxRate = 1.

# ΔT::Real, max_delay::Real, M::Real, offrate::Real, onrate::Real
Gate_Params = (Memory, 1, 10_000, 0, 1)
TI_Params = Gate_Params

unit = PulseIR.PoissonAutoNormalizedMultiply(
    [[Term_Denominator for _=1:Terms_Per_Product] for _=1:N_Products],
    base, TI_Threshold, Memory, Renorm_Rate, TI_Params, Gate_Params
)

function get_input_times(inrate, count)
    input_times = []
    t = 0
    while length(input_times) < count
        t += rand(Exponential(1/inrate))
        push!(input_times, t)
    end
    return input_times
end
get_count_inputs(inrate_per_line, in_vals) = [
        [
            (t, (:counts => prodidx => factoridx,))
            for t in get_input_times(inrate_per_line, term_n_inputs)
        ]
        for (prodidx, factor_invals) in enumerate(in_vals)
            for (factoridx, term_n_inputs) in enumerate(factor_invals)
    ] |> Iterators.flatten |> collect |> sort_by_time
sort_by_time(vals) = sort(vals, by=((t,_),) -> t)

in_vals = [
    [1/20, 2/20, 5/20],
    [3/20, 20/20, 1/20],
    [3/20, 5/20, 1/20]
]

count_inputs = get_count_inputs(Count_Inrate, Term_Denominator * in_vals)
events = Sim.simulate_for_time_and_get_events(
    implement_deep(unit, Spiking()),
    200,
    inputs=[
        count_inputs...,
        (last(count_inputs)[1] + 0.5, (:ind,))
    ]
)
dict = spiketrain_dict(
    filter(events) do (t, compname, event)
        compname === nothing
    end
)
getcnt(dict, key, mintime=0) = !haskey(dict, key) ? 0 : length([v for v in dict[key] if v ≥ mintime])

indouttime = only(dict["nothing: ind"])
@test indouttime > last(count_inputs)[1]
@test all(!haskey(dict, "nothing: :scaled_rates => $i") || last(dict["nothing: :scaled_rates => $i"]) < Memory for i=1:N_Products)

counts = [getcnt(dict, "nothing: :scaled_rates => $i", indouttime) for i=1:N_Products]
scale = getcnt(dict, "nothing: scale")

vals = [convert(Float64, base)^(-scale) * cnt / (Memory - indouttime) for cnt in counts]
product(vals) = reduce(*, vals)
fractional_errors = [(val - prod) / prod for (val, prod) in zip(vals, map(product, in_vals))]
@test all(abs(err) < 2. for err in fractional_errors)

function run_repeated_test(n_tests=100)
    eventss = []
    for i=1:n_tests
        push!(eventss, Sim.simulate_for_time_and_get_events(
            implement_deep(unit, Spiking()),
            200,
            inputs=[
                count_inputs...,
                (last(count_inputs)[1] + 0.5, (:ind,))
            ]
        )
        )
        println("Completed simulation $i")
    end
    fractional_errors = []
    for events in eventss
        dict = spiketrain_dict(
            filter(events) do (t, compname, event)
                compname === nothing && event isa Sim.OutputSpike
            end
        )
        
        indouttime = only(dict["nothing: ind"])
        @test indouttime > last(count_inputs)[1]
        @test all(!haskey(dict, "nothing: :scaled_rates => $i") || last(dict["nothing: :scaled_rates => $i"]) < Memory for i=1:N_Products)
        
        counts = [getcnt(dict, "nothing: :scaled_rates => $i", indouttime) for i=1:N_Products]
        scale = getcnt(dict, "nothing: scale")
        
        vals = [convert(Float64, base)^(-scale) * cnt / (Memory - indouttime) for cnt in counts]
        product(vals) = reduce(*, vals)
        push!(fractional_errors, [(val - prod) / prod for (val, prod) in zip(vals, map(product, in_vals))])
        @assert all(abs(err) < 4 for err in last(fractional_errors)) "$(last(fractional_errors))"
    end

    mean_errs = sum(fractional_errors)/length(fractional_errors)
    @test sum(mean_errs) / length(mean_errs) < 2/n_tests
    return events
end
run_repeated_test(20); nothing