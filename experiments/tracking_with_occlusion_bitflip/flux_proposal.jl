using DynamicModels
using Parameters: @with_kw
using Logging
using Statistics
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
# prior will almost always fail -- whenever a bit flips, prior will almost always predict it didn't.

# output a distribution of x, y ball and occluder. if its (2, 4) vs (4, 2) , don't want to propose 4, 4

include("model.jl")
include("groundtruth_rendering.jl")
include("prior_proposal.jl")
include("visualize.jl")
include("ann_utils.jl")


""" FLUX ANN TRAINING AND TESTING """ 

#loss(y, ŷ) = Flux.crossentropy(ŷ, y);

loss(y, ŷ) = Flux.kldivergence(y, ŷ, agg=sum)

#loss(y, ŷ) = Flux.mse(ŷ, y)

maxlen_lv_range = maximum(map(f-> length(f), lv_ranges))


#

# For KL, we are getting a value for each variable and summing. This operation is done on a matrix of values, with each row representing the digitized value of a latent variable. These functions makes a row for a matrix out of each variable and adds trailing zeros to ranges that are shorter than the max lv range.

parse_code_by_varb(y) = [vcat(Flux.softmax(convert(Vector{Float32}, y[i1+1:i2])), zeros(maxlen_lv_range-(i2-i1))) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]

parse_code_by_varb_no_zeros(y) = [Flux.softmax(convert(Vector{Float64}, y[i1+1:i2])) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]


""" NN TYPES, ARGS, AND TRAINING FUNCTIONS """ 

nn_single(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint), sum([length(l) for l in lv_ranges]), Flux.relu)) #, x-> parse_code_by_varb(x))

nn_one_hidden(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint),
               Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))), Flux.tanh),
    Flux.Dense(Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))),
               sum([length(l) for l in lv_ranges]), Flux.tanh))
#    Flux.softmax) #, x->parse_code_by_varb(x))

nn_two_hidden(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint),
               Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))), Flux.tanh),
    Flux.Dense(Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))), 
               Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))), Flux.tanh),
    Flux.Dense(Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))),
               sum([length(l) for l in lv_ranges]), Flux.tanh))



@with_kw mutable struct Args
    η = 3e-4             # learning rate (orig 3e-4)
    λ = 1e-4             # L2 regularizer param, implemented as weight decay
    batchsize = 5        # batch size
    epochs = 200  # number of epochs
    training_samples = 40
    validation_samples = 20
    num_smc_steps = 10
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
    save_every_n_epochs = epochs / 2   # Save the model every x epochs.
    tblogger = true       # log training with tensorboard
    savepath = "/Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion_bitflip/ann_logging/"
    model_name = "one_hidden_layer_meanloss"
end            


# you can do KL as an accuracy function. idea is you can call
# generate in the accuracy function based on the latents and obs. this should increase
# as you go. 

function eval_validation_set(data, model, device)
    total_loss = 0f0
    accuracy = 0
    for (prev_lv_and_img, gtlv) in data
        gt_curr_lv = parse_code_by_varb(gtlv)
        prev_lv_and_img, gt_curr_lv = prev_lv_and_img |> device, gt_curr_lv |> device
        ŷ = model(prev_lv_and_img)
        loss_input = hcat(parse_code_by_varb(ŷ)...), hcat(gt_curr_lv...)
        total_loss += loss(loss_input[1], loss_input[2])
        accuracy += sum(abs.(loss_input[2] - loss_input[1]))
        #        accuracy += sum([sum(y-g) for (y, g) in zip(ŷ, gt_curr_lv)])
#        accuracy += 0
    end
    return (loss = round(total_loss, digits=4), acc = round(accuracy, digits=4))
end


function train_nn_on_model(nn_args::Args, nn_generator::Function, input_encoder::Function)
    #  validation_data, training_data)
    vdr, validation_data = generate_training_data(nn_args.validation_samples, nn_args.num_smc_steps, input_encoder)
    nn_model = nn_generator(validation_data[1][1])
    modelpath = joinpath(nn_args.savepath,
                         string(nn_args.model_name, "nn_model.bson"))
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
#        run(`bash tensorboard --logdir /Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/ann_logging`, wait=false)
    end
    @info "Start Training"
    for epoch in 0:nn_args.epochs
        if nn_args.training_samples != 0
            tdr, training_data = generate_training_data(nn_args.training_samples, nn_args.num_smc_steps, input_encoder)
        else
            tdr = []
            training_data = []
        end
        # probably shuffle the training data here.
        println("Generated New Training Data")
        p = ProgressMeter.Progress(length(training_data))
        if epoch % nn_args.infotime == 0
            test_model_performance_on_v = eval_validation_set(
                validation_data, nn_model,device)
            test_model_performance_on_t = eval_validation_set(
                training_data, nn_model, device)
            println("Epoch: $epoch Validation: $(test_model_performance_on_v)")
            println("Epoch: $epoch Test: $(test_model_performance_on_t)")            
            if nn_args.tblogger
                set_step!(tblogger, epoch)
                with_logger(tblogger) do
                    @info "train" loss_validation=test_model_performance_on_v.loss acc_validation=test_model_performance_on_v.acc loss_test=test_model_performance_on_t.loss acc_test=test_model_performance_on_t.acc
            end
        end
        for (sample, groundtruth) in training_data
            sample, groundtruth = sample |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(sample)
                ŷ_tomod = convert(Matrix{Float32}, hcat(parse_code_by_varb(nn_model(sample))...))
                gt = convert(Matrix{Float32}, hcat(parse_code_by_varb(groundtruth)...))
                loss(ŷ_tomod, gt)
                #                loss(hcat(parse_code_by_varb(groundtruth)...), hcat(parse_code_by_varb(ŷ)...))
            end
            Flux.Optimise.update!(opt, nn_params, grads)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
                
        if epoch > 0 && epoch % nn_args.save_every_n_epochs == 0
            !ispath(nn_args.savepath) && mkpath(nn_args.savepath)
            let model=Flux.cpu(nn_model), nn_args=struct2dict(nn_args)
                BSON.@save modelpath nn_model epoch nn_args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    end
    return nn_model
end    


function load_ann(model_name)
    Core.eval(Main, :(import NNlib))
    b = BSON.@load string("/Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion_bitflip/saved_ann_models/", model_name, "nn_model.bson") nn_model epoch nn_args
    nn_model_w_softmax(x) = vcat(parse_code_by_varb_no_zeros(nn_model(x))...)
    return nn_model_w_softmax
end


# on each of step of this function, the latents and image from the
# groundtruth trace are used. 


#nn_symbolic_proposal = load_ann("one_hidden_layer_symbolic")
#nn_proposal = load_ann("one_hidden_layer")
nn_untrained_proposal = load_ann("one_hidden_layer_untrained")
nn_proposal = load_ann("one_hidden_layer_meanloss")

lv_no_info() = zeros(sum([length(l) for l in lv_ranges]))

# these proposals assume that velocity can be learned but it cant. you only have the previous velocity
# and the current image. there's no way to decode velocity from that, but it will be OK in the view of the model. 

@gen (static) function flux_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(nextstate_probs[end])
    xₜ ~ Cat(nextstate_probs[1])
    yₜ ~ Cat(nextstate_probs[2])
    vxₜ ~ VelCat(nextstate_probs[3])
    vyₜ ~ VelCat(nextstate_probs[4])
#    vxₜ ~ VelCat(onehot(xₜ - xₜ₋₁, Vels()))
#    vyₜ ~ VelCat(onehot(yₜ - yₜ₋₁, Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end


@gen (static) function flux_proposal_MAP(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(onehot(nextstate[end], OccPos()))

    # #NET CONTROLLING VELOCITY
    vxₜ ~ VelCat(onehot(nextstate[3], Vels()))
    vyₜ ~ VelCat(onehot(nextstate[4], Vels()))

    #DETERMINISTIC POSITION
    # xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, SqPos()))
    # yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, SqPos()))

    #NET CONTROLLING POSITION
    xₜ ~ Cat(onehot(nextstate[1], SqPos()))
    yₜ ~ Cat(onehot(nextstate[2], SqPos()))

    #DETERMINISTIC VELOCITY
#    vxₜ ~ VelCat(onehot(xₜ - xₜ₋₁, Vels()))
#    vyₜ ~ VelCat(onehot(yₜ - yₜ₋₁, Vels()))

    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end


@gen (static) function flux_untrained_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_untrained_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(nextstate_probs[end])
    xₜ ~ Cat(nextstate_probs[1])
    yₜ ~ Cat(nextstate_probs[2])
    vxₜ ~ VelCat(nextstate_probs[3])
    vyₜ ~ VelCat(nextstate_probs[4])
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end


@gen (static) function flux_untrained_proposal_MAP(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_untrained_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(onehot(nextstate[end], OccPos()))
    xₜ ~ Cat(onehot(nextstate[1], SqPos()))
    yₜ ~ Cat(onehot(nextstate[2], SqPos()))
    vxₜ ~ VelCat(onehot(nextstate[3], Vels()))
    vyₜ ~ VelCat(onehot(nextstate[4], Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end


# No option here of deterministic velocity or position -- have no previous latent info

@gen (static) function flux_initial_proposal(img)
    img_dig = image_digitize(img)
    no_latent_data_and_img_digitized = vcat(lv_no_info(), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_proposal(no_latent_data_and_img_digitized))
    occₜ ~ Cat(onehot(nextstate[end], OccPos()))
    xₜ ~ Cat(onehot(nextstate[1], SqPos()))
    yₜ ~ Cat(onehot(nextstate[2], SqPos()))
    vxₜ ~ VelCat(onehot(nextstate[3], Vels()))
    vyₜ ~ VelCat(onehot(nextstate[4], Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

# @gen (static function gen_trained_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
#     @param log_score_high::Float64
#     x_probs =       


# TODO 8/31: 
# compile the SNN for the 3D model and ask how many neurons there are.
# unrolled net will get the same obs but use its OWN latent calls from prevstates as the input
# not the traces calls.


# these are the last two images in the gt_tr and proposal. they are different. 

#example_image = gt_tr[DynamicModels.obs_addr(7)][1]
