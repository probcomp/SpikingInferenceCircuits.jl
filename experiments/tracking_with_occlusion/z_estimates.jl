function get_z_estimates()

end

function z_estimates_comparison(initial_traces)
    inferences = run_inference(initial_traces)
    gold_standard_z_estimates = get_gold_standard_z_estimates(inferences)
    z_estimates = get_z_estimates

end