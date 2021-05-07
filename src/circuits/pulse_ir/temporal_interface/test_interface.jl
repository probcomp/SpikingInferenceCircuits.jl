# Currently this file isn't being used.
# This is the start of some code to test that the
# temporal interfaces components is satisfied by their
# behavior in simulation.

produce_spiketrains(comp::Component, input_count_assignments,  int::CombinatoryInterface; n_cycles) =
    [
        let input_spiketrain = random_input_timings(input_count_assignments, int.input_windows)
            output_spiketrain = simulate_and_get_output_spiketrains(comp, input_spiketrain, total_length(int))
                (input, input_spiketrain, output_spiketrain)
        end
        for cycle=1:n_cycles
            for input in input_count_assignments
    ]

emperically_satisfies_holds(comp::Component, input_count_assignments, int::CombinatoryInterface; n_cycles = 4) =
    spiketrains_emperically_satisfy_holds(int.output_windows, (
        output_spiketrain for (_, _, output_spiketrain) in produce_spiketrains(comp, input_count_assignments, int; n_cycles)
    ))
spiketrains_emperically_satisfy_holds(windows, trains) = all(
    spiketrain_emperically_satisfies_holds(train, windows) for train in trains
)

test_deterministic_combinatory(comp::Component, input_count_assignments, int::CombinatoryInterface, fn; n_cycles=4) =
    let trains = produce_spiketrains(comp, input_count_assignments, int; n_cycles)
        (
            spiketrains_emperically_satisfy_holds(int.output_windows, trains) &&
            test_deterministic_io_values(trains, int, fn)
        )
    end
    
test_deterministic_io_values(trains, int, fn) = all(
        count(out_train, int.out_windows) = fn(in_assmt)
        for (in_assmt, intrain, out_train) in trains
    )

# TODO: tests for nondeterministic components

#=
Do we want to explicitely specify the valid input set and the value output sets,
and test that the outputs are always in the valid set?
=#