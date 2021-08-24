using DynamicModels
using Parameters: @with_kw
using Logging
import Flux
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay, ADAM
import CUDA
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter
using GLMakie
using Gen


# TODO

# write the digitize obs function according to vikash's answer about the
# structure of the training


# plot the distribution of xt when we see it and when we dont.
# condition on the fact that we don't see the ball and then hist x given occluder pos.

# xt vs obs given visible

# xt vs occluder_pos given dot invisible
# be flexible to x OR y. 

# occluder pos given occluder obs
# is occluded function is inside render_pixel


# xt on y axis xt-1 on x axis, but condition on v = 2 AND when you dont see the ball.
# flexible for velocity entry and flexible for whether you condition on ball being visible or not
# this is also a heatmap. 
# ALSO DO Y for every x plot you do
# do this all on the training data. 

include("model.jl")
include("groundtruth_rendering.jl")
include("prior_proposal.jl")
include("visualize.jl")
include("locally_optimal_proposal.jl")

ProbEstimates.use_perfect_weights!()
model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
@load_generated_functions()

lvs = [:xₜ, :yₜ, :vxₜ, :vyₜ, :occₜ]
lv_ranges = [SqPos(), SqPos(), Vels(), Vels(), OccPos()]
digitize(f) = f == Occluder() ? [0, 0, 1] : f == Empty() ? [0, 1, 0] : [1, 0, 0]
invert_digitized_obs(f) = f == [0, 0, 1] ? Occluder() : f == [0, 1, 0] ? Empty() : Object()
get_state_values(cmap) = [cmap[lv => :val] for lv in lvs]
lv_to_onehot_array(data_array) = vcat([onehot(d, hp) for (d, hp) in zip(data_array, lv_ranges)]...)
probs_to_lv_MAP(arr, lv_h) = sum(arr) == 0 ? lv_h[uniform_discrete(1, length(arr))] : lv_h[findmax(arr)[2]]
nn_probs_to_probs(arr) = sum(arr) == 0 ? normalize(ones(length(arr))) : normalize(arr)



# function you want for probabilistic proposal


""" DATA EXTRACTION FROM MENTAL PHYSICS TRACES """ 

function extract_image_array(cmap)
    img_array = []
    for row in 1:ImageSideLength()
        for col in 1:ImageSideLength()
            pix = digitize(cmap[:img_inner => row => col => :pixel_color => :val])
            img_array = vcat(img_array, pix)
        end
    end
    return img_array
end


function generate_training_data(num_trajectories::Int64, num_steps::Int64)
    training_data_raw = []
    training_data_digitized = []
    training_images = []
    gt_traces = []
    for traj in 1:num_trajectories
        tr = simulate(model, (num_steps,))
        push!(gt_traces, tr)
        for step in 1:num_steps
            td, tdr, image = digitize_trace(tr)
            push!(training_data_raw, tdr...)
            push!(training_data_digitized, td...)
            push!(training_images, image)
        end
    end
    return training_data_raw, training_data_digitized, training_images, gt_traces
end


function digitize_trace(tr)
    td_for_trace = []
    td_raw_for_trace = []
    images = []
    for step in 1:get_args(tr)[1]
        prevstate = get_state_values(latents_choicemap(tr, step-1))
        curr_image = extract_image_array(obs_choicemap(tr, step))
        # prev_image = extract_image_array(obs_choicemap(tr, step-1))
        # prev_ballpos = find_ball_location(prev_image)
        # curr_ballpos = find_ball_location(curr_image)
        # curr_velocity = curr_ballpos - prev_ballpos
        (image, ) = tr[DynamicModels.obs_addr(step)]
        push!(images, image)
        currstate = get_state_values(latents_choicemap(tr, step))
        training_datapoint_raw = (prevstate, curr_image, currstate)
        training_datapoint_digitized = [vcat(lv_to_onehot_array(training_datapoint_raw[1]), training_datapoint_raw[2]),
                                        lv_to_onehot_array(training_datapoint_raw[3])]
        push!(td_for_trace, training_datapoint_digitized)
        push!(td_raw_for_trace, training_datapoint_raw)
    end
    return td_for_trace, td_raw_for_trace, images
end



# have to generate tdr looking data from the ANN. idea is to generate a trace, then digitize it.
# put it through the net to get the answer and add that answer to the end of the list. do this a bunch of times

function make_dataarrays_from_trained_nn(nn_model_name, num_samples, num_steps)
    td_raw, td_dig, images, gt_traces = generate_training_data(num_steps, num_samples)
    nn_model = load_ann(nn_model_name)
    nn_lvs = [[tdr[1], tdr[2], extract_latents_from_nn(nn_model(td[1]))[1]] for (tdr, td) in zip(td_raw, td_dig)]
    return nn_lvs
end


function scatter_state_vs_prevstate(tdr)
    fig = Figure(resolution=(2000,500))
    axes = [Axis(fig[1,i]) for i in 1:length(lvs)]
    p_color = RGBAf0(0, 0, 0, float(20/length(tdr)))
    # index a color map here for different variables
    for lv in lvs
        data_ind = findfirst(f -> f == lv, lvs)
        plot_data = [(d[1][data_ind], d[3][data_ind]) for d in tdr]
        scatter!(axes[data_ind], plot_data, color=p_color)
#        heatmap!(axes[data_ind], plot_data)
        axes[data_ind].xlabel = string(lv, string(:₋₁))
        axes[data_ind].xlabelsize = 30
        axes[data_ind].ylabel = string(lv)
        axes[data_ind].ylabelsize = 30
    end
    display(fig)
end


function heatmap_state_vs_prevstate(tdr, lv)
    fig = Figure(resolution=(1000,1000))
    axes = Axis(fig[1,1]) 
    data_ind = findfirst(f -> f == lv, lvs)
    lv_hyp_array = lv_ranges[data_ind]
    hm_array = zeros(length(lv_hyp_array), length(lv_hyp_array))
    plot_data = [(d[1][data_ind], d[3][data_ind]) for d in tdr]
    for pd in plot_data
        prevstate = findfirst(f -> f == pd[1], lv_hyp_array)
        state = findfirst(f -> f == pd[2], lv_hyp_array)
        hm_array[prevstate, state] += 1
    end
    hm = heatmap!(axes, hm_array, colormap=:thermal)
    axes.xlabel = string(lv, string(:₋₁))
    axes.xlabelsize = 30
    axes.xticks = lv_hyp_array    
    axes.ylabel = string(lv)
    axes.ylabelsize = 30
    axes.yticks = lv_hyp_array
    cbar = Colorbar(fig[1,2], hm)
    display(fig)
end


function heatmap_lvs_in_current_state(tdr, lv1, lv2)
    fig = Figure(resolution=(1000,1000))
    axes = Axis(fig[1,1]) 
    data_ind_1 = findfirst(f -> f == lv1, lvs)
    data_ind_2 = findfirst(f -> f == lv2, lvs)
    lv_hyp_array_1 = lv_ranges[data_ind_1]
    lv_hyp_array_2 = lv_ranges[data_ind_2]
    hm_array = zeros(length(lv_hyp_array_1), length(lv_hyp_array_2))
    plot_data = [(d[3][data_ind_1], d[3][data_ind_2]) for d in tdr]
    for pd in plot_data
        state_v1 = findfirst(f -> f == pd[1], lv_hyp_array_1)
        state_v2 = findfirst(f -> f == pd[2], lv_hyp_array_2)
        hm_array[state_v1, state_v2] += 1
    end
    hm = heatmap!(axes, hm_array, colormap=:thermal)
    axes.xlabel = string(lv1)
    axes.xlabelsize = 30
    axes.xticks = lv_hyp_array_1    
    axes.ylabel = string(lv2)
    axes.ylabelsize = 30
    axes.yticks = lv_hyp_array_2
    cbar = Colorbar(fig[1,2], hm)
    display(fig)
end


# conditioner is going to be a mappable function over the
# zipped raw training data and training images
# f -> 

# here conditioner is an anonymous function called on either the image, the obs array, or the lvs
# common would be (f -> f[1][1][findfirst(x-> x == :xₜ, lvs] < 5)
# or  (f -> isnothing(find_ball_location(f[2])[1]))) for occlusion

function plot_conditioned_lv(tdr, training_images, conditioner)
    # start by zipping
    training_data_and_images = collect(zip(tdr, training_images))
    tdr_filtered_by_conditioner = [t[1] for t in filter(conditioner, training_data_and_images)]
end   


function make_lv_histogram(tdr, lv, state_or_prevstate)
    data_ind = findfirst(f -> f == lv, lvs)
    plot_data = [d[state_or_prevstate][data_ind] for d in tdr]
    fig = Figure()
    ax = Axis(fig[1,1])
    hist!(ax, plot_data, bins=length(lv_ranges[data_ind]))
    display(fig)
end

function find_ball_location(image)
    ball_x = get_x_pos(image)
    ball_y = isnothing(ball_x) ? nothing : get_y_pos(image, ball_x)
    return [ball_x, ball_y]
end



""" FLUX ANN TRAINING AND TESTING """ 



#loss(y, ŷ) = Flux.crossentropy(ŷ, y);
loss(y, ŷ) = Flux.mse(ŷ, y)


nn_cand1(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint), sum([length(l) for l in lv_ranges]), Flux.relu))

@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 5       # batch size
    epochs = 2           # number of epochs
    training_samples = 20
    validation_samples = 20
    num_smc_steps = 5
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
    save_every_n_epochs = epochs / 2   # Save the model every x epochs.
    tblogger = true       # log training with tensorboard
    savepath = "/Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/ann_logging/"
    model_name = "single_layer"
end            
                                        

function eval_validation_set(data, model, device)
    total_loss = 0f0
    accuracy = 0
    for (image, gt) in data
        image, gt = image |> device, gt |> device
        ŷ = model(image)
        total_loss += loss(ŷ, gt) 
        accuracy += sum([abs(diff) < .1 ? 1 : 0 for diff in ŷ-gt])
    end
    return (loss = round(total_loss, digits=4), acc = round(accuracy, digits=4))
end


function train_nn_on_model(nn_args::Args, nn_generator::Function)
    #  validation_data, training_data)
    vdr, validation_data = generate_training_data(nn_args.validation_samples, nn_args.num_smc_steps)
    nn_model = nn_generator(validation_data[1][1])
    println("Generated Validation Set")
    nn_args.seed > 0 && Random.seed!(nn_args.seed)
    use_cuda = nn_args.cuda && CUDA.has_cuda_gpu()
    if use_cuda
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end
    # shuffle here if you want to 
    nn_params = Flux.params(nn_model)
    opt = Optimiser(ADAM(nn_args.η), WeightDecay(nn_args.λ))
    if nn_args.tblogger 
        tblogger = TBLogger(nn_args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(nn_args.savepath)\""
        run(`bash tensorboard --logdir /Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/ann_logging`, wait=false)
    end
    @info "Start Training"
    for epoch in 0:nn_args.epochs
        # probably shuffle the training data here.
        tdr, training_data = generate_training_data(nn_args.training_samples, nn_args.num_smc_steps)
        println("Generated New Training Data")
        p = ProgressMeter.Progress(length(training_data))
        if epoch % nn_args.infotime == 0
            test_model_performance = eval_validation_set(
                validation_data, 
                nn_model,
                device)
            println("Epoch: $epoch Validation: $(test_model_performance)")            
            if nn_args.tblogger
                set_step!(tblogger, epoch)
                with_logger(tblogger) do
                    @info "train" loss=test_model_performance.loss acc=test_model_performance.acc
            end
                #            epoch == 0 && run(`bash tensorboard --logdir /Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/ann_logging/`, wait=false)
        end
        for (sample, groundtruth) in training_data
            sample, groundtruth = sample |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(sample)
                loss(ŷ, groundtruth)
            end
            Flux.Optimise.update!(opt, nn_params, grads)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
                
        if epoch > 0 && epoch % nn_args.save_every_n_epochs == 0
            !ispath(nn_args.savepath) && mkpath(nn_args.savepath)
            modelpath = joinpath(nn_args.savepath, string(nn_args.model_name, "nn_model.bson"))
            let model=Flux.cpu(nn_model), nn_args=struct2dict(nn_args)
                BSON.@save modelpath nn_model epoch nn_args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    end
    return nn_model
end    

    
function extract_latents_from_nn(digitized_array)
    latents = []
    latent_categorical_probs = []
    counter = 1
    # have to rearrange the values in latents to coincide w args to observable (occ, x, y, vx, vy)
    for lvh in lv_ranges
        digitized_lv = digitized_array[counter:counter+length(lvh)-1]
        extracted_value_for_lv = probs_to_lv_MAP(digitized_lv, lvh)
        push!(latents, extracted_value_for_lv)
        push!(latent_categorical_probs, nn_probs_to_probs(digitized_lv))
        counter += length(lvh)
    end
    (image, ) = obs_model(vcat(latents[end], latents[1:end-1])...)
    return latents, image, latent_categorical_probs
end


function load_ann(model_name)
    Core.eval(Main, :(import NNlib))
    b = BSON.@load string("./ann_logging/", model_name, "nn_model.bson") nn_model epoch nn_args
    return nn_model
end


function visualize_ann_answers(gt_trace, nn_type)
    nn_mod = load_ann(nn_type)
    length_smc = get_args(gt_trace)[1]
    trace_data_digitized = digitize_trace(gt_trace)[1]
    ann_answers = [extract_latents_from_nn(nn_mod(d[1])) for d in trace_data_digitized]
    (fig, t) = draw_obs(map(b->b[2], ann_answers))
    display(fig)
    animate(t, length_smc-1)
    make_video(fig, t, length_smc-1, "ann_obs.mp4")
    (fig, t) = draw_obs(gt_trace)
    make_video(fig, t, length_smc-1, "gt.mp4")
    return map(b->b[1], ann_answers)
end


occluded_bounce_constraints() = choicemap(
	(:init => :latents => :xₜ => :val, 1),
	(:init => :latents => :vxₜ => :val, 2),
    (:init => :latents => :occₜ => :val, 8),
    (:steps => 5 => :latents => :occₜ => :val, 8)
)

generate_occluded_bounce_tr() = generate(model, (15,), occluded_bounce_constraints())[1]
    


@gen (static) function flux_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
    img_dig = extract_image_array(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_mod(prevstate_and_img_digitized))
    occₜ ~ Cat(nextstate_probs[end])
    xₜ ~ Cat(nextstate_probs[1])
    yₜ ~ Cat(nextstate_probs[2])
    vxₜ ~ VelCat(nextstate_probs[3])
    vyₜ ~ VelCat(nextstate_probs[4], .2, Vels())
end

@gen (static) function flux_proposal_MAP(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
    img_dig = extract_image_array(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_mod(prevstate_and_img_digitized))
    occₜ ~ Cat(onehot(nextstate[end], OccPos()))
    xₜ ~ Cat(onehot(nextstate[1], SqPos()))
    yₜ ~ Cat(onehot(nextstate[2], SqPos()))
    vxₜ ~ VelCat(onehot(nextstate[3], Vels()))
    vyₜ ~ VelCat(onehot(nextstate[4], Vels()))
end


# TODO 8/24: test plotting functions. get ANN training onto GPU. test whether

    # NN unrolling. where does this come in conceptually? idea will be to
    # start with an initial state and an observation. can unroll the entire SMC from there with the NN recursively
