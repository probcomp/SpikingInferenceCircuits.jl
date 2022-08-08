using Circuits, SpikingCircuits, Serialization
const Sim = SpikingCircuits.SpikingSimulator
includet("../utils/spiketrain_utils.jl")

# Run hyperparams:
save_file() = "snn_runs/better_organized/velwalk1d/10timesteps200interval/2021-07-26__02-02"
snn_n_particles() = 2;
inter_obs_interval() = 200.
ISWeightDenominator() = 200 # multiplier weight denominator used for this particular run
n_steps() = 10

# Get events
# events = (@time deserialize(save_file()));

### These will be useful for getting the exact product of the estimates:
# get_particle_trains(t) = [
#     get_sample_score_recip_labels_trains(events, i, collect(pairs((xₜ=Positions(), vₜ=Vels()))), t, inter_obs_interval())
#     for i=1:snn_n_particles()
# ]

# function score_estimates(t)
#     trains = get_particle_trains(t)
#     return [
#         (fwd_score_estimates(), recip_score_estimates())
#     ]
# end

particle_addr(t, particle) =
        if t == 1
            :initial_step => :particles => particle
        else
            :subsequent_steps => :smcstep => :particles => particle
        end

function events_for_step(events, t)
    starttime = inter_obs_interval() * (t - 1)
    endtime = starttime + inter_obs_interval()
    return events_in_timerange(events, starttime, endtime)
end

### Output of multiplier:
num_multiplier_output_spikes(events, t, particle) = 
    filter(events_for_step(events, t)) do (_, c, e)
        c == particle_addr(t, particle) && e isa Sim.OutputSpike && e.name == (:weight => :count)
    end |> length
weight_estimate(events, t, particle) = 
    num_multiplier_output_spikes(events, t, particle) / ISWeightDenominator()

weight_estimates(events) = [weight_estimate(events, t, particle) for particle=1:snn_n_particles(), t=1:n_steps()]

vel_to_val(v_indexed) = Vels()[v_indexed]
function prev_smc_state_and_obs(events, t, particle)
    evts = filter(events_for_step(events, t)) do (t, c, e)
        c == particle_addr(t, particle) && e isa Sim.InputSpike
    end
    evt_names = [e.name for (_, _, e) in evts]
    args = [n.second for n in evt_names if n.first == :args]
    v_indexed = only([n.second for n in args if n.first == :vₜ₋₁])
    x = only([n.second for n in args if n.first == :xₜ₋₁])
    obs = only([n.second.second for n in evt_names if n.first == :yᵈₜ])
    return (x, vel_to_val(v_indexed), obs)
end
prev_smc_states_and_obs(events) =
    [
        prev_smc_state_and_obs(events, t, particle)
        for particle=1:snn_n_particles(), t=2:n_steps()
    ]

function smc_state(events, t, particle)
    # filter(events_for_step(events, t)) do (_, c, e); c == particle_addr(1, 1) && e isa Sim.OutputSpike && e.name.first != :weight; end
    evts = filter(events_for_step(events, t)) do (_, c, e)
        c == particle_addr(t, particle) && e isa Sim.OutputSpike && e.name.first != :weight
    end
    evt_names = [e.name for (_, _, e) in evts]
    trace = [n.second for n in evt_names if n.first == :trace]
    x = only([n.second for n in trace if n.first == :xₜ])
    v = only([n.second for n in trace if n.first == :vₜ])
    return (xₜ=x, vₜ=vel_to_val(v))
end

smc_states(events) = [smc_state(events, t, particle) for particle=1:snn_n_particles(), t=2:n_steps()]

logweight_for_transition((prevx, prevv, obs), (xₜ, vₜ)) = ProbEstimates.with_weight_type(:perfect) do
    latents_ch = choicemap((:xₜ => :val, xₜ), (:vₜ => :val, vₜ))
    propscore, _ = assess(_approx_step_proposal, (prevx, prevv, obs), latents_ch)
    assess_latents_score, _ = assess(step_latent_model, (prevx, prevv), latents_ch)
    assess_obs_score, _ = assess(obs_model, (xₜ, yₜ), choicemap((:yᵈₜ => :val, obs)))
    assess_score = assess_latents_score + assess_obs_score
    assess_score - propscore
end
get_true_logweights_for_snn_states(events) = [
    logweight_for_transition((prevx, prevv, obs), (xₜ, vₜ))
        for ((prevx, prevv, obs), (xₜ, vₜ)) in zip(
            prev_smc_states_and_obs(events), # prev states for time=2:end
            smc_states(events) # current states for time 2:end
        )
    ]

function true_to_snn_logweights(events)
    trueweights = get_true_logweights_for_snn_states(events)
    noisy_weights = weight_estimates(events)[2:end]
    true_to_noisy = Dict{Float64, Vector{Float64}}()
    for (true_weight, noisy_weight) in zip(trueweights, noisy_weights)
        push!(get!(true_to_noisy, true_weight, []), log(noisy_weight))
    end
    return true_to_noisy
end

dict = true_to_snn_logweights(events)