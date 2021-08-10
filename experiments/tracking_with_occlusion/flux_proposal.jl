using DynamicModels
using Flux
using Flux: crossentropy, logitcrossentropy, mse, onecold
using Parameters: @with_kw
using Logging
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay, ADAM
using DelimitedFiles
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter

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
    for traj in 1:num_trajectories
        tr = simulate(model, (num_steps,))
        for step in 1:num_steps
            prevstate = get_state_values(latents_choicemap(tr, step-1))
            obs_image = extract_image_array(obs_choicemap(tr, step))
            currstate = get_state_values(latents_choicemap(tr, step))
            push!(training_data_raw, (prevstate, obs_image, currstate))
        end
    end
    training_data_digitized = [(lv_to_onehot_array(d[1]), d[2],
                                lv_to_onehot_array(d[3])) for d in training_data_raw]
    return training_data_raw, training_data_digitized
end

function plot_state_vs_prevstate(tdr)
    fig = Figure(resolution=(1000,500))
    axes = [Axis(fig[1,i]) for i in 1:length(lvs)]
    # index a color map here for different variables
    for lv in lvs
        data_ind = findfirst(f -> f == lv, lvs)
        plot_data = [(d[1][data_ind], d[3][data_ind]) for d in tdr]
        scatter!(axes[data_ind], plot_data)
        axes[data_ind].xlabel = string(lv)
    end
    display(fig)
end


function train_nn_on_dataset(nn_model::Chain, nn_args::Args,
                             validation_set, training_set)
    nn_args.seed > 0 && Random.seed!(nn_args.seed)
    use_cuda = nn_args.cuda && CUDAapi.has_cuda_gpu()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    validation_loader = DataLoader(
        validation_set...,
        batchsize=1)
    training_loader = DataLoader(
        training_set...,
        batchsize=nn_args.batchsize)
    nn_params = params(nn_model)
    opt = Optimiser(ADAM(nn_args.η), WeightDecay(nn_args.λ))

    if nn_args.tblogger 
        tblogger = TBLogger(nn_args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(nn_args.savepath)\""
    end
    
    @info "Start Training"
    for epoch in 0:nn_args.epochs
        p = ProgressMeter.Progress(length(training_loader))
        if epoch % nn_args.infotime == 0
            test_model_performance = eval_validation_set(
                validation_loader, 
                nn_model,
                device)
            println("Epoch: $epoch Validation: $(test_model_performance)")            
            if nn_args.tblogger
                set_step!(tblogger, epoch)
                with_logger(tblogger) do
                    @info "train" loss=test_model_performance.loss acc=test_model_performance.acc
            end
            epoch == 0 && run(`tensorboard --logdir logging`, wait=false)
        end
        for (sample, groundtruth) in training_loader
            sample, groundtruth = sample |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(sample)
                loss(ŷ, groundtruth[1, :, 1, :])
            end
            Flux.Optimise.update!(opt, nn_params, grads)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
                
        if epoch > 0 && epoch % nn_args.save_every_n_epochs == 0
            !ispath(nn_args.savepath) && mkpath(nn_args.savepath)
            modelpath = joinpath(nn_args.savepath, "nn_model.bson") 
            let model=cpu(nn_model), nn_args=struct2dict(nn_args)
                BSON.@save modelpath nn_model epoch nn_args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    end
    return nn_model
end    
    
    
            
   



