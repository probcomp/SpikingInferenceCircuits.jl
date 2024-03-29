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

lvs = [:xₜ, :yₜ, :vxₜ, :vyₜ, :occₜ]
lv_ranges = [SqPos(), SqPos(), Vels(), Vels(), OccPos()]
lv_ranges_symbolic = [SqPos(), SqPos(), OccPos()]
digitize(f) = f == Occluder() ? [0, 0, 1] : f == Empty() ? [0, 1, 0] : f == Object() ? [1, 0, 0] : Nothing
invert_digitized_obs(f) = f == [0, 0, 1] ? Occluder() : f == [0, 1, 0] ? Empty() : Object()
get_state_values(cmap) = [cmap[lv => :val] for lv in lvs]
lv_to_onehot_array(data_array, lvrs) = vcat([isnothing(d) ? zeros(length(hp)) : onehot(d, hp) for (d, hp) in zip(data_array, lvrs)]...)
probs_to_lv_MAP(arr, lv_h) = lv_h[findmax(arr)[2]]
#nn_probs_to_probs(arr) = sum(arr) == 0 ? normalize(ones(length(arr))) : normalize(arr)
#sliding_window(arr) = zip(arr[1:end-1], arr[2:end])
sliding_window(arr) = [[arr[i], arr[i+1]] for i in 1:length(arr)-1]
my_cumsum(arr) = map(x-> sum(arr[1:x+1]), 0:length(arr)-1)


""" DATA EXTRACTION FROM MENTAL PHYSICS TRACES """

# there are two ways of getting an image. visualize_ann and visualize_unrolling
# render an image from a latent variable set. i extract the image array from the
# choicemap of the gt_trace and input it to the NN in these cases. the NN outputs a
# latent variable. the image array goes row by row and extracts the values along the columns.
# the image that comes into the proposal is opposite. it contains the information in each
# column and you are digitizing that way. therefore the image is off.


function image_digitize_by_row(img)
    img_array = []
    for row in 1:ImageSideLength()
        pix = [digitize(p[row]) for p in img]
        img_array = vcat(img_array, pix...)
    end
    return img_array
end

function image_transpose(img)
    img_matrix = []
    for row in 1:ImageSideLength()
        pix = [p[row] for p in img]
        img_matrix = push!(img_matrix, pix)
    end
    return img_matrix
end

function extract_image_array(cmap)
    img_array = []
    for row in 1:ImageSideLength()
        for col in 1:ImageSideLength()
            pix = digitize(cmap[:img_inner => row => col => :pixel_color => :color => :val])
            img_array = vcat(img_array, pix)
        end
    end
    return img_array
end


# input encoder will either be "digitize_trace" or "digitize_trace_symbolic"

function generate_training_data(num_trajectories::Int64, num_steps::Int64, input_encoder::Function)
    training_data_raw = []
    training_data_digitized = []
    training_images = []
    gt_traces = []
    for traj in 1:num_trajectories
        tr = simulate(model, (num_steps,))
        push!(gt_traces, tr)
        td, tdr, image = input_encoder(tr)
        push!(training_data_raw, tdr...)
        push!(training_data_digitized, td...)
        push!(training_images, image)
    end
    return training_data_raw, training_data_digitized, training_images, gt_traces
end

function digitize_trace_symbolic(tr)
    td_for_trace = []
    td_raw_for_trace = []
    images = []
    for step in 1:get_args(tr)[1]
        prevstate = get_state_values(latents_choicemap(tr, step-1))
        currstate = get_state_values(latents_choicemap(tr, step))
        (image, ) = tr[DynamicModels.obs_addr(step)]
        curr_ballpos = find_ball_location(image_transpose(image))
        curr_occpos = get_occluder_pos(image_transpose(image))
        td_raw = [curr_ballpos[1], curr_ballpos[2], curr_occpos]
        td_symbolic_dig = vcat(lv_to_onehot_array(prevstate, lv_ranges),
                               lv_to_onehot_array(td_raw, lv_ranges_symbolic))
        push!(td_raw_for_trace, (prevstate, td_raw, currstate))
        push!(td_for_trace, [td_symbolic_dig, lv_to_onehot_array(currstate, lv_ranges)])
        push!(images, image)
    end
    return td_for_trace, td_raw_for_trace, images
end
    
function digitize_trace(tr)
    td_for_trace = []
    td_raw_for_trace = []
    images = []
    for step in 1:get_args(tr)[1]
        prevstate = get_state_values(latents_choicemap(tr, step-1))
        curr_image = extract_image_array(obs_choicemap(tr, step))
        (image, ) = tr[DynamicModels.obs_addr(step)]
        push!(images, image)
        currstate = get_state_values(latents_choicemap(tr, step))
        training_datapoint_raw = (prevstate, curr_image, currstate)
        training_datapoint_digitized = [vcat(lv_to_onehot_array(training_datapoint_raw[1], lv_ranges),
                                             training_datapoint_raw[2]),
                                        lv_to_onehot_array(training_datapoint_raw[3], lv_ranges)]
        push!(td_for_trace, training_datapoint_digitized)
        push!(td_raw_for_trace, training_datapoint_raw)
    end
    return td_for_trace, td_raw_for_trace, images
end



# have to generate tdr looking data from the ANN. idea is to generate a trace, then digitize it.
# put it through the net to get the answer and add that answer to the end of the list. do this a bunch of times

function make_dataarrays_from_trained_nn(nn_model_name, num_samples, num_steps, input_encoder)
    td_raw, td_dig, images, gt_traces = generate_training_data(num_steps, num_samples, input_encoder)
    nn_model = load_ann(nn_model_name)
    nn_lvs = [[tdr[1], tdr[2], extract_latents_from_nn(nn_model(td[1]))[1]] for (tdr, td) in zip(td_raw, td_dig)]
    return [td_raw, td_dig], nn_lvs
end

function make_dataarrays_from_trained_nn(td, nn_model_name)
    td_raw, td_dig = td
    nn_model = load_ann(nn_model_name)
    nn_lvs = [[tdr[1], tdr[2], extract_latents_from_nn(nn_model(td[1]))[1]] for (tdr, td) in zip(td_raw, td_dig)]
    return nn_lvs
end

    
function create_random_encoding(num_steps, nn_model_name)
    nn_model = load_ann(nn_model_name)
    lv_encoding = [vcat([onehot(uniform_discrete(lvr[1], lvr[end]), lvr) for lvr in lv_ranges]...) for i in 1:num_steps]
    image_encoding = [vcat([[[1, 0, 0], [0, 1, 0], [0, 0, 1]][uniform_discrete(1,3)]
                            for j in 1:length(SqPos())^2]...) for i in 1:num_steps]
    println(image_encoding)
    training_data = [vcat(b...) for b in zip(lv_encoding, image_encoding)]
    # extract to latents to get training_data_raw. then pass this to make_data_arrays w two args
    return training_data
end




function scatter_state_vs_prevstate(tdr)
    fig = Figure(resolution=(2000,500))
    axes = [Axis(fig[1,i]) for i in 1:length(lvs)]
    p_color = RGBAf0(0, 0, 0, float(20/length(tdr)))
    # index a color map here for different variables
    for (data_ind, lv) in enumerate(lvs)
#        data_ind = findfirst(f -> f == lv, lvs)
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


# vy has two datapoints in groundtruth!! and y

# ANSWERS ARE GENERALY UNIFORM. 


""" FLUX ANN TRAINING AND TESTING """ 

# cross entropy didn't work at all. 
#loss(y, ŷ) = Flux.crossentropy(ŷ, y);

loss(y, ŷ) = Flux.kldivergence(y, ŷ, agg=sum)

#loss(y, ŷ) = Flux.mse(ŷ, y)

# REPRESENT EACH Y AS A MATRIX INSTEAD OF A VECTOR

maxlen_lv_range = maximum(map(f-> length(f), lv_ranges))



# this makes a row for a matrix out of each variable and adds trailing zeros to ranges that are shorter than the max lv range. 
parse_code_by_varb(y) = [vcat(Flux.softmax(convert(Vector{Float32}, y[i1+1:i2])), zeros(maxlen_lv_range-(i2-i1))) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]

parse_code_by_varb_no_zeros(y) = [Flux.softmax(convert(Vector{Float64}, y[i1+1:i2])) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]


#parse_code_by_varb(y) = [vcat(convert(Vector{Float32}, y[i1+1:i2]), zeros(maxlen_lv_range-(i2-i1))) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]

nn_single(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint), sum([length(l) for l in lv_ranges]), Flux.relu)) #, x-> parse_code_by_varb(x))

nn_one_hidden(input_datapoint) = Flux.Chain(
    Flux.Dense(length(input_datapoint),
               Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))), Flux.tanh),
    Flux.Dense(Int(round(mean([length(input_datapoint), sum([length(l) for l in lv_ranges])]))),
               sum([length(l) for l in lv_ranges]), Flux.tanh))
#    Flux.softmax) #, x->parse_code_by_varb(x))

#nn_one_hidden(input_datapoint) = Flux.Chain(Flux.Dense(length(input_datapoint), length(input_datapoint), Flux.tanh), Flux.Dense(length(input_datapoint), sum([length(l) for l in lv_ranges]), Flux.tanh), Flux.softmax) #, x->parse_code_by_varb(x))

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
    model_name = "one_hidden_layer"
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
        tdr, training_data = generate_training_data(nn_args.training_samples, nn_args.num_smc_steps, input_encoder)
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

    
function extract_latents_from_nn(digitized_array)
    latents = []
    latent_categorical_probs = []
    counter = 1
    # have to rearrange the values in latents to coincide w args to observable (occ, x, y, vx, vy)
    for lvh in lv_ranges
        digitized_lv = digitized_array[counter:counter+length(lvh)-1]
        extracted_value_for_lv = probs_to_lv_MAP(digitized_lv, lvh)
        push!(latents, extracted_value_for_lv)
        push!(latent_categorical_probs, digitized_lv)
        counter += length(lvh)
    end
    (image, ) = obs_model(vcat(latents[end], latents[1:end-1])...)
    return latents, image, latent_categorical_probs
end


function load_ann(model_name)
    Core.eval(Main, :(import NNlib))
    dir_name = joinpath(
        Base.find_package("SpikingInferenceCircuits") |> dirname |> dirname,
        "experiments/tracking_with_occlusion/proposals/saved_ann_models/"
    )
    b = BSON.@load string(dir_name, model_name, "nn_model.bson") nn_model epoch nn_args
    nn_model_w_softmax(x) = vcat(parse_code_by_varb_no_zeros(nn_model(x))...)
    return nn_model_w_softmax
end


# on each of step of this function, the latents and image from the
# groundtruth trace are used. 

function visualize_ann_answers(gt_trace, nn_type, input_encoder)
    nn_mod = load_ann(nn_type)
    length_smc = get_args(gt_trace)[1]
    trace_data_digitized = input_encoder(gt_trace)[1]
    ann_answers = [extract_latents_from_nn(nn_mod(d[1])) for d in trace_data_digitized]
    make_comparison_vids(ann_answers, gt_trace)
    return map(b->b[1], ann_answers)
end

function make_comparison_vids(ann_answers, gt_trace)
    length_smc = get_args(gt_trace)[1]
    (fig2, t2) = draw_obs(gt_trace)
    display(fig2)
    animate(t2, length_smc-1)
    make_video(fig2, t2, length_smc-1, "gt.mp4")
    (fig, t) = draw_obs(map(b->b[2], ann_answers))
    display(fig)
    animate(t, length_smc-1)
    make_video(fig, t, length_smc-1, "ann_obs.mp4")
end
    
function unroll_nn(trace_digitized, nn_mod, latents_state_t, lvranges_for_unrolling)
    # takes the obs and the previous nn_answer.
    if isempty(trace_digitized)
       return latents_state_t
    end
    nn_input = vcat(latents_state_t[end],
                    first(trace_digitized)[1][sum(map(x->length(x), lvranges_for_unrolling))+1:end])
    push!(latents_state_t, nn_mod(nn_input))
    unroll_nn(trace_digitized[2:end], nn_mod, latents_state_t, lvranges_for_unrolling)
end


# on each step of this function, only the previous NN answers for
# latent variables and the current image are used. 

function visualize_unrolled_ann(gt_trace, nn_type::String, input_encoder::Function, lvrs)
    nn_mod = load_ann(nn_type)
    trace_data_digitized = input_encoder(gt_trace)[1]
    unrolled_latent_calls_dig = unroll_nn(trace_data_digitized, nn_mod,
                                          [nn_mod(trace_data_digitized[1][1])], lvrs)
    unrolled_latent_calls = [extract_latents_from_nn(nn_answer) for nn_answer in unrolled_latent_calls_dig]
    make_comparison_vids(unrolled_latent_calls, gt_trace)
    return unrolled_latent_calls
end


function unit_test_encodings(n_steps)
    training_data_raw, training_data_digitized, training_images, gt_traces = generate_training_data(1, n_steps)
    # here want the training_data_raw inputs (grabbed from the choicemap) and the training_data_digitized inputs.
    # extract_latents_from_nn should be applied to the digitized inputs. the image extracted from the obs_model should match
    # the training_images index which is just the image extracted from the trace.
    data_comparison = []
    for i in 1:n_steps
        gt_image = training_images[1][i]
        gt_latents = training_data_raw[i][3]
        latents_extracted_from_encoded, ims, lcprobs = extract_latents_from_nn(training_data_digitized[i][2])
        # this will give you the image from the previous step, not the current step. 
        image_from_decoded_data = obs_model(vcat(latents_extracted_from_encoded[end], latents_extracted_from_encoded[1:end-1])...)
#        return gt_image, image_from_decoded_data, gt_latents, latents_extracted_from_encoded
        push!(data_comparison, gt_latents == latents_extracted_from_encoded && [g for g in gt_image] == [f for f in image_from_decoded_data...]) # &&
    end
    return data_comparison
end

#    return gt_image, gt_latents, latents_extracted_from_encoded, image_from_decoded_data


#nn_symbolic_proposal = load_ann("one_hidden_layer_symbolic")
nn_proposal = load_ann("one_hidden_layer")


image_digitize(img) = vcat([digitize(impix) for impix in vcat(img...)]...)


# these proposals assume that velocity can be learned but it cant. you only have the previous velocity
# and the current image. there's no way to decode velocity from that, but it will be OK in the view of the model. 

@gen (static) function _flux_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(nextstate_probs[end])
    vxₜ ~ VelCat(nextstate_probs[3])
    vyₜ ~ VelCat(nextstate_probs[4])
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, positions(SquareSideLength())))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, positions(SquareSideLength())))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end


@gen (static) function _flux_proposal_MAP(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
#    img_dig = image_digitize_by_row(img)
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate, img, nextstate_probs = extract_latents_from_nn(nn_proposal(prevstate_and_img_digitized))
    occₜ ~ Cat(onehot(nextstate[end], OccPos()))
    # xₜ ~ Cat(onehot(nextstate[1], SqPos()))
    # yₜ ~ Cat(onehot(nextstate[2], SqPos()))
    vxₜ ~ VelCat(onehot(nextstate[3], Vels()))
    vyₜ ~ VelCat(onehot(nextstate[4], Vels()))
    xₜ ~ Cat(onehot(xₜ₋₁ + vxₜ, positions(SquareSideLength())))
    yₜ ~ Cat(onehot(yₜ₋₁ + vyₜ, positions(SquareSideLength())))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)    
end

flux_proposal = @compile_step_proposal(_flux_proposal, obs_aux_proposal, 5, 1)
flux_proposal_MAP = @compile_step_proposal(_flux_proposal_MAP, obs_aux_proposal, 5, 1)