include("model.jl")

ProbEstimates.use_perfect_weights!()
model = @DynamicModel(init_latent_model, step_latent_model, obs_model, 5)
@load_generated_functions()

lvs = [:xₜ, :yₜ, :vxₜ, :vyₜ, :occₜ]
lv_ranges = [SqPos(), SqPos(), Vels(), Vels(), OccPos()]
lv_ranges_symbolic = [SqPos(), SqPos(), OccPos()]
digitize(f) = f == Occluder() ? [0, 0, 1] : f == Empty() ? [0, 1, 0] : f == Object() ? [1, 0, 0] : Nothing
invert_digitized_obs(f) = f == [0, 0, 1] ? Occluder() : f == [0, 1, 0] ? Empty() : Object()
get_state_values(cmap) = [cmap[lv => :val] for lv in lvs]
lv_to_onehot_array(data_array, lvrs) = vcat([isnothing(d) ? zeros(length(hp)) : onehot(d, hp) for (d, hp) in zip(data_array, lvrs)]...)
probs_to_lv_MAP(arr, lv_h) = lv_h[findmax(arr)[2]]
#nn_probs_to_probs(arr) = sum(arr) == 0 ? normalize(ones(length(arr))) : normalize(arr)
sliding_window(arr) = [[arr[i], arr[i+1]] for i in 1:length(arr)-1]
my_cumsum(arr) = map(x-> sum(arr[1:x+1]), 0:length(arr)-1)
image_digitize(img) = vcat([digitize(impix) for impix in vcat(img...)]...)

function image_digitize(img::List{Any})
    image_digitized = []
    for i in 1:length(img)
        im_row = img[i]
        for j in 1:length(im_row)
            push!(image_digitized, digitize(im_row[j]))
        end
    end
    return vcat(image_digitized...)
end


""" DATA EXTRACTION FROM MENTAL PHYSICS TRACES """

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




""" VISUALIZATION OF ANN ANSWERS: CAN USE WITH TORCH OR FLUX MODELS """


function visualize_ann_answers(gt_trace, nn_mod, input_encoder)
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

function visualize_unrolled_ann(gt_trace, nn_mod, input_encoder::Function, lvrs)
    trace_data_digitized = input_encoder(gt_trace)[1]
    unrolled_latent_calls_dig = unroll_nn(trace_data_digitized, nn_mod,
                                          [nn_mod(trace_data_digitized[1][1])], lvrs)
    unrolled_latent_calls = [extract_latents_from_nn(nn_answer) for nn_answer in unrolled_latent_calls_dig]
    make_comparison_vids(unrolled_latent_calls, gt_trace)
    return unrolled_latent_calls
end


""" VALIDATION OF ANN ACCURACY """


function make_dataarrays_from_trained_nn(nn_model, num_samples, num_steps, input_encoder)
    td_raw, td_dig, images, gt_traces = generate_training_data(num_steps, num_samples, input_encoder)
    nn_lvs = [[tdr[1], tdr[2], extract_latents_from_nn(nn_model(td[1]))[1]] for (tdr, td) in zip(td_raw, td_dig)]
    return [td_raw, td_dig], nn_lvs
end

function make_dataarrays_from_trained_nn(td, nn_model)
    td_raw, td_dig = td
    nn_lvs = [[tdr[1], tdr[2], extract_latents_from_nn(nn_model(td[1]))[1]] for (tdr, td) in zip(td_raw, td_dig)]
    return nn_lvs
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


function make_lv_histogram(tdr, lv, state_or_prevstate)
    data_ind = findfirst(f -> f == lv, lvs)
    plot_data = [d[state_or_prevstate][data_ind] for d in tdr]
    fig = Figure()
    ax = Axis(fig[1,1])
    hist!(ax, plot_data, bins=length(lv_ranges[data_ind]))
    display(fig)
end


""" UNIT TESTING OF ENCODINGS """ 



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




