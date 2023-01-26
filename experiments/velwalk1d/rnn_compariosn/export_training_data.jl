# Export training data into a JSON file to import in python.
# (I intend to eventually improve the Julia <> Python interfacing here
# so PyTorch can pull training examples directly from the model.)

import JSON
includet("main.jl")

function dataset_object(n_datapoints, rollout_length)
    return Dict(
        "traces" => [
            trace_to_dict(simulate(model, (rollout_length,)))
            for _=1:n_datapoints
        ],
        "Positions" => collect(Positions()),
        "Vels" => collect(Vels()),
        "ObsStd" => ObsStd(),
        "VelStepStd" => VelStepStd()
    )
end

function trace_to_dict(tr)
    return [
        Dict(
            "x" => latents_choicemap(tr, t)[:xₜ => :val],
            "v" => latents_choicemap(tr, t)[:vₜ => :val],
            "obs" => obs_choicemap(tr, t)[:obs => :val]
        )
        for t=1:get_args(tr)[1]
    ]
end

open("training_data.json", "w") do io
    JSON.print(io, dataset_object(10000, 10))
end