include("logging_utils.jl")
using Base.Filesystem
import Dates
using Serialization

function get_save_path(experiment_dir)
    savedir = joinpath(experiment_dir, "saves")
    if !isdir(savedir)
        mkdir(savedir)
    end
    
    # find a filename which isn't already taken!
    time = Dates.format(Dates.now(), "yyyy-mm-dd__HH-MM")
    save = 1
    filename = joinpath(savedir, time)
    while isfile(filename)
        save += 1
        filename = joinpath(savedir, time*"_save$save")
    end

    return filename
end

function simulate_and_get_events(
    implemented_circuit,
    time_to_simulate_for,
    inputs;
    log_interval=400, # events
    save_events=true,
    # by default, save runs in the `experiments/saves` folder;
    # can also accept another folder giving the location of the experiment being run
    dir=joinpath(
        #  src/SpikingInferenceCircuits.jl              src/         
        Base.find_package("SpikingInferenceCircuits") |> dirname |> dirname,
        "experiments"
    )
)
    savepath = save_events ? get_save_path(dir) : nothing
    if save_events
        println("Will save resulting events at $savepath")
    end
    println("Beginning simulation.")
    events = try
        SpikingSimulator.simulate_for_time_and_get_events(
            implemented_circuit,
            time_to_simulate_for;
            inputs,
            log=true,
            log_filter=get_log_filter(log_interval),
            log_str=time_log_str
        )
    catch e
        @error "Error $e arose while running simulation!"
    end
    println("Simulation completed.")
    
    if save_events
        println("Saving resulting events at $savepath...")
        try
            serialize(savepath, events)
        catch e
            @error "Error $e arose while serializing events from simulation!"
        end
    end

    return events
end

# latents and observations should be indexed (not labeled)
# TODO: support labels
function get_smc_circuit_inputs(
    time_to_simulate_for,
    interval_between_observations,
    observations
)
    first_obs, remaining_obs = Iterators.peel(observations)
    inputs = Tuple{Float64, Tuple}[
        (
            0.,
            (
                (:obs => key => val for (key, val) in pairs(first_obs))...,
                :is_initial_obs
            )
        )
    ]
    t = 0
    while t < time_to_simulate_for
        t += interval_between_observations
        obs, remaining_obs = Iterators.peel(remaining_obs)
        push!(inputs, 
            (t, ((:obs => key => val for (key, val) in pairs(obs))...,))
        )
    end
    return inputs
end

# latents and observations should be indexed (not labeled)
function get_smc_circuit_inputs_with_initial_latents(
    time_to_simulate_for,
    interval_between_observations,
    initial_latents,
    observations,
    nparticles
)
    obs_inputs = get_smc_circuit_inputs(
        time_to_simulate_for,
        interval_between_observations,
        observations
    )
    first_input, rest = Iterators.peel(obs_inputs)
    @assert first_input[1] == 0.
    return [
        (0., (
            (
                :initial_latents => i => key => val
                for (key, val) in pairs(initial_latents)
                    for i=1:nparticles
            )...,
            (input for input in first_input[2] if input != :is_initial_obs)...
        )),
        rest...
    ]
end