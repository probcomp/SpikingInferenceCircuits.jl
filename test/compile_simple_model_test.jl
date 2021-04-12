include("spiketrain_analysis.jl")

### implementation rules ###

REF_RATE() = 1.0
OFF_RATE() = 0.0001
ON_RATE() = 2.0 * REF_RATE()

Circuits.implement(::SIC.PositiveReal, ::Spiking) =
    SIC.SpikeRateReal(REF_RATE())
Circuits.implement(p::SIC.PositiveRealMultiplier, ::Spiking) =
    SIC.RateMultiplier(
        6.0, REF_RATE(),
        Tuple(SIC.SpikeRateReal(REF_RATE()) for _=1:p.n_inputs)
    )
Circuits.implement(c::SIC.CPTSampleScore, ::Spiking) =
    SIC.SpikingCPTSampleScore(c, OFF_RATE(), ON_RATE())

# simple test model
@gen (static) function test(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    z ~ CPT([
        [[0.9, 0.1]] [[0.1, 0.9]];
        [[0.75, 0.25]] [[0.25, 0.75]]
    ])(x, y)
    return z
end
@load_generated_functions()

### PROPOSE TESTS ###
RUN_LENGTH() = 10.0
mean(vec) = sum(vec) / length(vec)

# Get circuit for `test` in `PROPOSE` mode
# We must tell it the number of possible values that `input` can take (here, 2)
propose_circuit = gen_fn_circuit(test, (input=FiniteDomain(2),), Propose())
propose_implemented = implement_deep(propose_circuit, Spiking())

propose_events1 = [
    SpikingSimulator.simulate_for_time_and_get_events(propose_implemented, RUN_LENGTH();
        initial_inputs=(:inputs => :input => 1,)
    ) for _=1:100
] |> output_spiketrain_dict
kl1 = propose_emperical_kl_to_true(test, (1,), propose_events1)
println("KL1 : $kl1")
@test kl1 < 5.0

propose1_prob_comparisons = [
    compare_spike_rate_weight(test, (1,), Propose(), propose_events1, RUN_LENGTH())
]
propose1_biases = [a - b for (a, b) in propose1_prob_comparisons]
@test mean(propose1_biases) < 0.2

propose_events2 = [
    SpikingSimulator.simulate_for_time_and_get_events(propose_implemented, RUN_LENGTH();
        initial_inputs=(:inputs => :input => 2,)
    ) for _=1:100
] |> output_spiketrain_dict
kl2 = propose_emperical_kl_to_true(test, (1,), propose_events2)
println("KL2 : $kl2")
@test kl2 < 5.0

propose2_rate_comparisons = [
    compare_spike_rate_weight(test, (2,), Propose(), propose_events2, RUN_LENGTH())
]
propose2_biases = [a - b for (a, b) in propose1_prob_comparisons]
@test mean(propose2_biases) < 0.2

### GENERATE TESTS ###
# gen fn circuit for `generate(test, args, choicemap(:y=..., :z=...)`
# ie. sample a value for `x`, and output `P_ancestral[y, z | x]`
generate_circuit = gen_fn_circuit(test, (input=FiniteDomain(2),), Generate(select(:y, :z)))
generate_implemented = implement_deep(generate_circuit, Spiking())

# ancestral distribution we should have sampled from:
@gen (static) function x_ancestral(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
end
@load_generated_functions()

obs1 = choicemap((:y, 1), (:z, 2))
generate_events_1 = [
    SpikingSimulator.simulate_for_time_and_get_events(generate_implemented, 20.0;
        initial_inputs=(:inputs => :input => 1, :obs => :y => obs1[:y], :obs => :z => obs1[:z])
    ) for _=1:20
]
gen_kl1 = propose_emperical_kl_to_true(x_ancestral, (1,), generate_events_1)
@test gen_kl1 < 1.0

gen1_weight_comparisons = [
    compare_spike_rate_weight(
        test, (1,), Generate(select(:x, :y)),
        generate_events_1, RUN_LENGTH();
        obs=obs1
    )
]
gen1_biases = [a - b for (a, b) in gen1_weight_comparisons]
@test mean(gen1_biases) < 0.2

obs2 = choicemap((:y, 2), (:z, 2))
generate_events_2 = [
    SpikingSimulator.simulate_for_time_and_get_events(generate_implemented, 20.0;
        initial_inputs=(:inputs => :input => 2, :obs => :y => obs2[:y], :obs => :z => obs2[:z])
    ) for _=1:20
]
gen_kl2 = propose_emperical_kl_to_true(generate_events_2, (1,), generate_events_2)
@test gen_kl2 < 1.0

gen2_weight_comparisons = [
    compare_spike_rate_weight(
        test, (2,), Generate(select(:x, :y)),
        generate_events_2, RUN_LENGTH();
        obs=obs2
    )
]
gen2_biases = [a - b for (a, b) in gen2_weight_comparisons]
@test mean(gen2_biases) < 0.2