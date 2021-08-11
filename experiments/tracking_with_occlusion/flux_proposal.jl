using DynamicModels
using Parameters: @with_kw
using Logging
import Flux
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay, ADAM
using DelimitedFiles
import CUDA
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter

# TODO
# make plots a heatmap instead of a scatter

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
lv_hps = [positions(SquareSideLength()), positions(SquareSideLength()),
          Vels(), Vels(), positions(OccluderLength())]
digitize(f) = f == Occluder() ? [0, 0, 1] : f == Empty() ? [0, 1, 0] : [1, 0, 0]
get_state_values(cmap) = [cmap[lv => :val] for lv in lvs]
lv_to_onehot_array(data_array) = vcat([onehot(d, hp) for (d, hp) in zip(data_array, lv_hps)]...)



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


function generate_samples(num_trajectories::Int64, num_steps::Int64)
    training_data_raw = []
    training_images = []
    for traj in 1:num_trajectories
        tr = simulate(model, (num_steps,))
        for step in 1:num_steps
            prevstate = get_state_values(latents_choicemap(tr, step-1))
            obs_image = extract_image_array(obs_choicemap(tr, step))
            (image, ) = tr[DynamicModels.obs_addr(step)]
            currstate = get_state_values(latents_choicemap(tr, step))
            push!(training_data_raw, (prevstate, obs_image, currstate))
            push!(training_images, image)
        end
    end
    training_data_digitized = [[vcat(lv_to_onehot_array(d[1]), d[2]),
                                lv_to_onehot_array(d[3])] for d in training_data_raw]
    return training_data_raw, training_data_digitized, training_images
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
    lv_hyp_array = lv_hps[data_ind]
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


function find_ball_location(image)
    ball_x = get_x_pos(image)
    ball_y = isnothing(ball_x) ? :Nothing : get_y_pos(image, ball_x)
    return ball_x, ball_y
end

    

""" FLUX ANN TRAINING AND TESTING """ 



#loss(y, ŷ) = Flux.crossentropy(ŷ, y);
loss(y, ŷ) = Flux.mse(ŷ, y)


nn_cand1(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint), sum([length(l) for l in lv_hps]), Flux.relu))

@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 5       # batch size
    epochs = 10           # number of epochs
    training_samples = 200
    validation_samples = 20
    num_smc_steps = 20
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
    save_every_n_epochs = epochs / 2   # Save the model every x epochs.
    tblogger = true       # log training with tensorboard
    savepath = "/Users/nightcrawler2/SpikingInferenceCircuits.jl/experiments/tracking_with_occlusion/ann_logging/"
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


function train_nn_on_dataset(nn_model::Flux.Chain, nn_args::Args,
                             validation_data, training_data)
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
    end
    
    @info "Start Training"
    for epoch in 0:nn_args.epochs
        # probably shuffle the training data here. 
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
            epoch == 0 && run(`tensorboard --logdir /Users/nightcrawler2/SpikingInferenceCircuits/experiments/tracking_with_occlusion/ann_logging`, wait=false)
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
            modelpath = joinpath(nn_args.savepath, "nn_model.bson") 
            let model=Flux.cpu(nn_model), nn_args=struct2dict(nn_args)
                BSON.@save modelpath nn_model epoch nn_args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    end
    return nn_model
end    
    

function flux_wrapper(nn_args::Args)
    tdr, td = generate_samples(nn_args.training_samples, nn_args.num_smc_steps)
    println("Generated Training Data")
    vdr, vd = generate_samples(nn_args.validation_samples, nn_args.num_smc_steps)
    println("Generated Validation Set")
    nn_model = nn_cand1(td[1][1])
    train_nn_on_dataset(nn_model,
                        nn_args,
                        vd, td)
end

    
                   
                   
    
    
    
   



