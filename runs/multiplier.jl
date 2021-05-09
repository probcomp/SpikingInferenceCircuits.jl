using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

in_denoms = (2., 3.)
out_denom = 5.
mult = PulseIR.PoissonSpikeCountMultiplier(
    in_denoms,
    out_denom,
    10,
    50.,
    100.,
    0.5,
    ((500, 12), 0.), #(ti_params, offrate)
    (500, 12)
)

println("Mult constructed.")

implemented = implement_deep(mult, Spiking())

println("Mult implemented deeply.")

### Simulation ###
include("spiketrain_utils.jl")

get_events() = SpikingSimulator.simulate_for_time_and_get_events(
    implemented, 500., initial_inputs=(
        :counts => 1, :counts => 1, :counts => 1, :counts => 1,
        :counts => 2, :counts => 2,
        :ind
    ) # 2 * 2/3 = 4/3
    # so should get 5 * 4/3 spikes out
    # ie. on average should get 20/3 = 6.666...
)
events = get_events()

println("Simulation completed.")

draw_fig(events)

# Do a bunch of runs and check whether the expected count looks right.

dicts = [out_st_dict(get_events()) for _=1:1500]
bad_apples = findall([!haskey(d, :ind) for d in dicts])

for i in reverse(bad_apples)
    deleteat!(dicts, i)
end

sum(haskey(d, :count) ? length(d[:count]) : 0 for d in dicts) / length(dicts)

#=
This experiment seems to indicate that the circuit is overshooting and usually
outputs too many spikes?

But it may just be that the issue causing the existance of "bad apples"
tends to occur for lower spike counts, which is biasing this estimation!

Fixing the "bad apples" is probably first priority.
=#