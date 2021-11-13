using ProbEstimates: AutonormalizeCountThreshold, AutonormalizeSpeedupFactor, AutonormalizeRepeaterRate
using Distributions

struct AutonormalizationData
    log_normalization_line::Vector{Float64}
    normalized_weight_lines::Vector{Vector{Float64}}
end

function get_time_when_scores_all_ready(is_spiketrain_data)
    all_ready_times = Iterators.flatten(
        [
            dense_val_train.ready_time
            for dense_val_train in [values(datum.fwd_trains)..., values(datum.recip_trains)...]
        ]
        for datum in is_spiketrain_data
    )

    maximum(all_ready_times)
end

# This didn't seem to align with what was happening in Gen particle filtering, so I'm just
# gonna go with Gen particle filtering values for now.  Eventually I'm curious to debug this.
# Not sure why this didn't work.
# """
# Returns a vector giving the log score update for each particle
# (ie. the log weight calculated at *this step* for the particle).
# """
# function get_log_score_updates(is_spiketrain_data)
#     [
#         log_score_update(datum)
#         for datum in is_spiketrain_data
#     ]
# end
# function log_score_update(datum::ISSpiketrains)
#     log_product_of_dense_values(values(datum.fwd_trains), K_fwd()) + log_product_of_dense_values(values(datum.recip_trains), K_recip())
# end
# function log_product_of_dense_values(dense_val_trains, denominator)
#     [
#         log(
#             sum(
#                 length(times_for_neuron)
#                 for times_for_neuron in dense_val_train.neuron_times
#             ) / denominator
#         )
#         for dense_val_train in dense_val_trains
#     ] |> sum
# end

function produce_autonormalization_spiketrains(
    starttime, unnormalized_log_values;
    n_spikes_to_accumulate_before_ending_autonormalization,
    speedup_factor,
    autonormalization_repeater_rate,
    total_simulation_time
)
    normalization_spiketimes = []
    particle_spiketimes = [[] for _ in unnormalized_log_values]
    
    rates = exp.(unnormalized_log_values)
    if sum(rates) == 0
        return AutonormalizationData(
            normalization_spiketimes,
            particle_spiketimes
        )
    end

    num_accumulated_spikes = 0
    current_time = starttime

    n_iters_in_loop = 0

    while num_accumulated_spikes < n_spikes_to_accumulate_before_ending_autonormalization
        time_to_repeater = rand(Exponential(1/autonormalization_repeater_rate))
        spikes_before_then = rand(Poisson(sum(rates) * time_to_repeater))
        total_this_would_accumulate_to = num_accumulated_spikes + spikes_before_then

        _add_spikes_to_autonormalization!(particle_spiketimes, rates, current_time, time_to_repeater, spikes_before_then, starttime + total_simulation_time)
        current_time += time_to_repeater

        if total_this_would_accumulate_to < n_spikes_to_accumulate_before_ending_autonormalization
            num_accumulated_spikes = total_this_would_accumulate_to
            push!(normalization_spiketimes, current_time)

            rates = rates .* speedup_factor
        else
            num_accumulated_spikes = n_spikes_to_accumulate_before_ending_autonormalization
        end

        # for debugging infinite loops:
        n_iters_in_loop += 1
        if n_iters_in_loop > 10^6
            error("n_iters_in_loop = $n_iters_in_loop")
        end
    end

    time_left_in_simulation = total_simulation_time - (current_time - starttime)

    if time_left_in_simulation > 0
        n_spikes_before_end_of_simulation = rand(Poisson(sum(rates) * time_left_in_simulation))
        _add_spikes_to_autonormalization!(particle_spiketimes, rates, current_time, time_left_in_simulation,
            n_spikes_before_end_of_simulation,
            starttime + total_simulation_time
        )
    end

    return AutonormalizationData(
        normalization_spiketimes,
        particle_spiketimes
    )
end
function _add_spikes_to_autonormalization!(
    particle_spiketimes, rates,
    current_time, time_window_to_add_spikes_in_length,
    number_of_spikes_to_add,
    time_after_which_not_to_add_spikes
)
    if current_time ≥ time_after_which_not_to_add_spikes
        return;
    end
    # Poisson process spikes are uniformly distributed within an interval, given the number of spikes in that interval
    spiketimes = sort([rand(Uniform(current_time, current_time + time_window_to_add_spikes_in_length)) for _=1:number_of_spikes_to_add])
    for spiketime in spiketimes
        if spiketime ≥ time_after_which_not_to_add_spikes
            break;
        else
            idx = rand(Categorical(rates / sum(rates)))
            push!(particle_spiketimes[idx], spiketime)
        end
    end
end

function get_autonormalization_data(
    is_spiketrain_data,
    other_factors_to_multiply_in_during_autonormalization;
    n_spikes_to_accumulate_before_ending_autonormalization=AutonormalizeCountThreshold(),
    speedup_factor=AutonormalizeSpeedupFactor(),
    autonormalization_repeater_rate=AutonormalizeRepeaterRate(),
    total_weight_readout_time=Latency(),
    expected_log_weight_updates
)
    autonorm_starttime = get_time_when_scores_all_ready(is_spiketrain_data)
    # log_score_updates = get_log_score_updates(is_spiketrain_data)::Vector{<:Real}

    log_score_updates = expected_log_weight_updates
    log_unnormalized_weights = log_score_updates .+ log.(other_factors_to_multiply_in_during_autonormalization::Vector{<:Real})

    # if isnothing(expected_log_weight_updates)
    #     @warn "Currently not checking log scores in NG-F spiketrain generator from Gen particle weights!"
    # else
    #     @assert isapprox(log_score_updates, expected_log_weight_updates) "log_score_updates = $log_score_updates ; expected_log_weight_updates = $expected_log_weight_updates"
    # end

    return produce_autonormalization_spiketrains(
        autonorm_starttime,
        log_unnormalized_weights;
        n_spikes_to_accumulate_before_ending_autonormalization,
        speedup_factor, autonormalization_repeater_rate,
        total_simulation_time=total_weight_readout_time
    )
end