includet("../proposals/obs_aux_proposal.jl")
###
function set_ngf_hyperparams_for_aux!(hyperparams)
    ProbEstimates.set_latency!(hyperparams.latency)
    # ProbEstimates.set_maxrate!(hyperparams.frequency)
    @assert ProbEstimates.MaxRate() == hyperparams.frequency
    ProbEstimates.set_minprob!(min(√(ColorFlipProb()), ProbEstimates.DefaultMinProb()))

    #=
    We need to distribute the neurons between
    1. Model - flip1 - 2 assemblies
    2. Model - flip2 - 2 assemblies
    3. Model - color - 3 assemblies
    4. Proposal - flip1 - 2 assemblies
    5. Proposal - flip2 - 2 assemblies

    Total: 11 assemblies
    =#
    ProbEstimates.set_assembly_size!(floor(hyperparams.neuron_budget / 11))
end
function set_ngf_hyperparams_for_noaux!(hyperparams)
    # ProbEstimates.set_maxrate!(hyperparams.frequency)
    @assert ProbEstimates.MaxRate() == hyperparams.frequency
    ProbEstimates.set_latency!(hyperparams.latency)
    ProbEstimates.set_minprob!(min(ColorFlipProb(), ProbEstimates.DefaultMinProb()))
    #=
    We just need to have neurons for `color`, which uses 3 assemblies.
    =#
    ProbEstimates.set_assembly_size!(floor(hyperparams.neuron_budget / 3))
end

function get_k(hyperparams; use_aux)
    if use_aux
        neurons_per_assembly = hyperparams.neuron_budget / 11
    else
        neurons_per_assembly = hyperparams.neuron_budget / 3
    end
    hyperparams.latency * neurons_per_assembly * hyperparams.frequency
end
### 
function color_dist(color)
    p = ColorFlipProb()
    return [
        c == color ? (1 - p) : p/2
        for c in PixelColors()
    ]
end
@gen (static) function maybe_flip_color_noaux(color)
    color ~ LCat(PixelColors())(color_dist(color))
    return color
end

@gen (static) function fill_in_color(input_color, output_color)
    color ~ LCat(PixelColors())(onehot(output_color, PixelColors()))
end

@gen (static) function fill_in_flips_color(input_color, output_color)
    is_correct = input_color == output_color

    # invalid deterministic proposal
    # flip1 ~ BoolCat(bern_probs(is_correct ? 0. : 1.0))
    # flip2 ~ BoolCat(bern_probs(is_correct ? 0. : 1.0))

    # exact proposal
    flip1 ~ BoolCat(bern_probs(
        is_correct ? prob_flip1_given_no_flip() : 1.0
    ))
    flip2 ~ BoolCat(bern_probs(
        is_correct ? prob_flip2_given_no_flip(flip1) : 1.0)
    )

    # 
    # flip1 ~ BoolCat(bern_probs(is_correct ? 1/3 : 1.0))
    # flip2 ~ BoolCat(bern_probs(is_correct ? (flip1 ? 0.0 : 1/2) : 1.0))

    color ~ LCat(PixelColors())(onehot(output_color, PixelColors()))
end

@load_generated_functions()

###

empirical_expectation(v) = sum(v) / length(v)
function empirical_variance_of_log_values(logvalues)
    # vs = exp.(logvalues)
    # empirical_expectation([x^2 for x in vs]) - empirical_expectation(vs)^2

    log_expectation = logsumexp(logvalues) - log(length(logvalues))
    log_sum_of_values_squared = logsumexp([lv * 2 for lv in logvalues])
    log_E_val_squared = log_sum_of_values_squared - log(length(logvalues))

    log_expectation_squared = log_expectation * 2
    return exp(log_E_val_squared) - exp(log_expectation_squared)

    # expectation = exp(logexpectation)
    # return sum(exp(lv)^2 for lv in logvalues) - expectation^2
    #     (exp(logvalue) - expectation)^2 for logvalue in logvalues
    # ) / length(logvalues)
end
function get_empirical_fracvar(; is_flipped, hyperparams, n_samples, use_aux)
    color_flip_model = use_aux ? maybe_flip_color : maybe_flip_color_noaux
    proposal = use_aux ? fill_in_flips_color : fill_in_color
    if use_aux
        set_ngf_hyperparams_for_aux!(hyperparams)
    else
        set_ngf_hyperparams_for_noaux!(hyperparams)
    end

    # TODO: change the global hyperparameters depending
    # whether we are or are not using aux vars

    model_args = (Empty(),)
    proposal_args = (Empty(), is_flipped ? Occluder() : Empty())
    
    (_, log_normalized_weights, log_ml_estimate) =
    ProbEstimates.with_weight_type(
        :noisy,
        () -> Gen.importance_sampling(
            color_flip_model, model_args,
            EmptyChoiceMap(),
            proposal, proposal_args,
            n_samples
        )
    )
    log_total_weight = log_ml_estimate + log(n_samples)
    log_weights = log_normalized_weights .+ log_total_weight
    # display(log_weights)

    true_score = is_flipped ? ColorFlipProb() : (1 - ColorFlipProb())
    emp_var = empirical_variance_of_log_values(log_weights)
    # println("emp_var: $emp_var")
    # println("emp fractional var: $(emp_var / true_score^2)")
    return emp_var/true_score^2
end

function get_empirical_fracvars(hyperparams, n_samples, use_aux)
    (
        get_empirical_fracvar(; is_flipped=true, hyperparams, use_aux, n_samples),
        get_empirical_fracvar(; is_flipped=false, hyperparams, use_aux, n_samples)
    )
end

indep_product_empirical_fracvar(fracvars) =
    prod((fv + 1) for fv in fracvars) - 1

function image_likelihood_fracvars(
    hyperparams, sidelength, n_flips_to_compute_var_for, use_aux, n_samples_for_single_pixel_estimate
)
    (flip_fracvar, noflip_fracvar) = get_empirical_fracvars(hyperparams, n_samples_for_single_pixel_estimate, use_aux)
    println("Flip fracvar: $flip_fracvar | NoFlip fracvar: $noflip_fracvar")
    return [
        indep_product_empirical_fracvar([i ≤ n_flips ? flip_fracvar : noflip_fracvar for i=1:sidelength^2])   
        for n_flips in n_flips_to_compute_var_for
    ]
end
    