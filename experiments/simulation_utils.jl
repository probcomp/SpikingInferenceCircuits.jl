include("logging_utils.jl")

simulate_and_get_events(
    implemented_circuit,
    time_to_simulate_for,
    inputs;
    log_interval=400 # events
) = SpikingSimulator.simulate_for_time_and_get_events(
    implemented_circuit,
    time_to_simulate_for;
    inputs,
    log=true,
    log_filter=get_log_filter(log_interval),
    log_str=time_log_str
)

# latents and observations should be indexed (not labeled)
function get_smc_circuit_inputs(
    time_to_simulate_for,
    interval_between_observations,
    initial_latents,
    observations,
    nparticles
)
    first_obs, remaining_obs = Iterators.peel(observations)
    inputs = Tuple{Float64, Tuple}[
        (0., (
            (
                :initial_latents => i => key => val
                for (key, val) in pairs(initial_latents)
                    for i=1:nparticles
            )...,
            (
                :obs => key => val
                for (key, val) in pairs(first_obs)
            )...
        ))
    ]
    t = 0
    while t < time_to_simulate_for
        t += interval_between_observations
        obs, remaining_obs = Iterators.peel(observations)
        push!(inputs, 
            (t, ((
                :obs => key => val
                for (key, val) in pairs(obs)
            )...,))
        )
    end
    return inputs
end