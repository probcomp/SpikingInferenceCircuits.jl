using PyCall
using GenPyTorch
using Statistics
using Gen
using JLD
using PyCallJLD
import Flux: softmax
import Base: zero

include("model.jl")
include("modeling_utils.jl")
include("groundtruth_rendering.jl")
include("visualize.jl")
include("obs_aux_proposal.jl")
include("prior_proposal.jl")
include("nearly_locally_optimal_proposal.jl")
include("run_utils.jl")
include("ann_utils.jl")


torch = pyimport("torch")
nn = torch.nn
F = nn.functional

# NOTE THESE DONT HAVE TO BE DEFINED IF USING DYNAMIC W CATEGORICALS or DYNAMIC W CAT, VELCAT 
#zero(t::NTuple{5, <:Any}) = (0.0, 0.0, 0.0, 0.0, 0.0)
#zero(t::NTuple{5, <:Any}) = nothing
#Gen.accumulate_param_gradients!(trace) = Gen.accumulate_param_gradients!(trace, nothing)

# NOTE:

macro Name(arg)
   string(arg)
end

# init_param! also zeros the gradient; but this is done anyway if you want to do more training. 
function Gen.set_param!(gf::TorchGenerativeFunction, name::String, value)
    gf.params[name] = value
end


tdr, td, t_im, gt_trs = generate_training_data(1, 15, digitize_trace)
input_dp = convert(Vector{Float64}, td[1][1])
input_dp_pos = vcat(input_dp[1:length(SqPos())*2], input_dp[length(SqPos())*2 + length(Vels())*2 + 1:end])
input_dp_image = input_dp[(length(input_dp) - 3*length(SqPos())^2) + 1:end]
maxlen_lv_range = maximum(map(f-> length(f), lv_ranges))
partition_nn_output(y) = [softmax(y[i1+1:i2]) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]
lv_no_info() = zeros(sum([length(l) for l in lv_ranges]))
lv_pos_no_info() = zeros(sum([length(l) for l in lv_ranges_symbolic]))



""" PYTHON DEFINITIONS FOR TORCH NETS """ 


@pydef mutable struct SingleHidden <: nn.Module
    function __init__(self, input_dp)
        # Note the use of pybuiltin(:super): built in Python functions
        # like `super` or `str` or `slice` are all accessed using
        # `pybuiltin`.
        pybuiltin(:super)(SingleHidden, self).__init__()
        self.dense1 = nn.Linear(
            length(input_dp),
            Int(round(mean([length(input_dp), sum([length(l) for l in lv_ranges])]))))
        self.dense2 = nn.Linear(
            Int(round(mean([length(input_dp), sum([length(l) for l in lv_ranges])]))),
            sum([length(l) for l in lv_ranges]))
    end

    function forward(self, x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    end

    function num_flat_features(self, x)
        # Note: x.size() returns a tuple, not a tensor.
        # Therefore, we treat it like a Julia tuple and
        # index using 1-based indexing.
        size = x.size()[2:end]
        num_features = 1
        for s in size
            num_features *= s
        end
        return num_features
    end
end


@pydef mutable struct TwoHidden <: nn.Module
    function __init__(self, input_dp, output_ranges)
        # Note the use of pybuiltin(:super): built in Python functions
        # like `super` or `str` or `slice` are all accessed using
        # `pybuiltin`.
        pybuiltin(:super)(TwoHidden, self).__init__()
        self.dense1 = nn.Linear(
            length(input_dp),
            Int(round(mean([length(input_dp), sum([length(l) for l in output_ranges])]))))

        self.dense2 = nn.Linear(
            Int(round(mean([length(input_dp), sum([length(l) for l in output_ranges])]))), 
            Int(round(mean([length(input_dp), sum([length(l) for l in output_ranges])]))))

        self.dense3 = nn.Linear(
            Int(round(mean([length(input_dp), sum([length(l) for l in output_ranges])]))),
            sum([length(l) for l in output_ranges]))
    end

    function forward(self, x)
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.dense2(x))
        x = F.tanh(self.dense3(x))
        return x
    end

    function num_flat_features(self, x)
        # Note: x.size() returns a tuple, not a tensor.
        # Therefore, we treat it like a Julia tuple and
        # index using 1-based indexing.
        size = x.size()[2:end]
        num_features = 1
        for s in size
            num_features *= s
        end
        return num_features
    end
end


# figure out the datatype of what alex is adding here and then accomodate. 

@pydef mutable struct ConvNet <: nn.Module
    function __init__(self, input_dp)
        pybuiltin(:super)(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    end

    function forward(self, x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    end
end


""" UTILITIES FOR GENERATING TRAINING DATA AND TRAINING TORCH NETS """ 

#nn_mod = SingleHidden(input_dp)
nn_mod_full = TwoHidden(input_dp, lv_ranges)
nn_torchgen_full = TorchGenerativeFunction(nn_mod_full, [TorchArg(true, torch.float)], 1)

nn_mod_pos = TwoHidden(input_dp_pos, lv_ranges_symbolic)
nn_torchgen_pos = TorchGenerativeFunction(nn_mod_pos, [TorchArg(true, torch.float)], 1)

nn_mod_image = TwoHidden(input_dp_image, lv_ranges_symbolic)
nn_torchgen_image = TorchGenerativeFunction(nn_mod_image, [TorchArg(true, torch.float)], 1)


# in example, measurements is the input to the proposal.
# construct this by sampling a one step model each time and properly assigning
# the previous states and image to the model. 

function groundtruth_generator()
    step = 2
    (tr, w) = generate(model, (2,))
    # construct arguments to the proposal function being trained.
    # for you its goint to be occ, x, y, vx, vy, img
    (image, ) = tr[DynamicModels.obs_addr(step)]
    prevstate = get_state_values(latents_choicemap(tr, step-1))
    currstate = get_state_values(latents_choicemap(tr, step))
    inputs = (prevstate[end], prevstate[1], prevstate[2], prevstate[3], prevstate[4], image)
    # construct constraints for the proposal function being trained
    constraints = Gen.choicemap()
    constraints[:occₜ => :val] = currstate[end]
    constraints[:xₜ => :val] = currstate[1]
    constraints[:yₜ => :val] = currstate[2]
    constraints[:vxₜ => :val] = currstate[3]
    constraints[:vyₜ => :val] = currstate[4]
    return (inputs, constraints)
end;



function train_torch_nn(torchfunction, proposal, filename)
    parameter_update = Gen.ParamUpdate(Gen.ADAM(0.001, 0.9, 0.999, 1e-8), 
                                       torchfunction => collect(get_params(torchfunction)))
    scores = Gen.train!(proposal, groundtruth_generator, parameter_update,
                        num_epoch=100, epoch_size=100, num_minibatch=100, minibatch_size=100,
                        evaluation_size=10, verbose=true);
    params_post_training = Dict()
    for k in keys(torchfunction.params)
        params_post_training[k] = torchfunction.params[k]
    end
    params_post_training["scores"] = scores
    PyCallJLD.save(string("saved_ann_models/", filename, ".jld"), "params_post_training", params_post_training)
    f = Figure()
    ax = Axis(f[1,1])
    lines!(ax, scores)
    display(f)
    return params_post_training
end

    

function load_torch_nn(torchfunction, filename)
    params_post_training = PyCallJLD.load(string("saved_ann_models/", filename, ".jld"),
                                          "params_post_training")
    [Gen.set_param!(torchfunction, name, params_post_training[name]) for name in keys(
         params_post_training) if name != "scores"]
    f = Figure()
    ax = Axis(f[1,1])
    scores = params_post_training["scores"]
    lines!(ax, scores)
    display(f)
end


# next steps here are to use the 6.885 pset as a guide for loading and saving the pytorch params for
# each run. 

    # if you're at velocity 1 OR 2 you'll crash into the wall at position 9.


                   


""" CUSTOM PROPOSALS FOR SMC """ 
    
@gen function torch_proposal_image(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    img_dig = image_digitize(img)
    nextstate_probs ~ nn_torchgen_image(img_dig)
    occₜ ~ Cat(softmax(nextstate_probs[21:28]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(pos_to_vel_dist(xₜ, xₜ₋₁))
    vyₜ ~ VelCat(pos_to_vel_dist(yₜ, yₜ₋₁))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@gen function torch_initial_proposal_image(img)
    img_dig = image_digitize(img)
    nextstate_probs ~ nn_torchgen_image(img_dig)
    occₜ ~ Cat(softmax(nextstate_probs[21:28]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(uniform(Vels()))
    vyₜ ~ VelCat(uniform(Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end


@gen function torch_proposal_full(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate_probs ~ nn_torchgen_full(prevstate_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[31:38]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(softmax(nextstate_probs[21:25]))
    vyₜ ~ VelCat(softmax(nextstate_probs[26:30]))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@gen function torch_initial_proposal_full(img)
    img_dig = image_digitize(img)
    no_latent_data_and_img_digitized = vcat(lv_no_info(), img_dig)
    nextstate_probs ~ nn_torchgen_full(no_latent_data_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[31:38]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(softmax(nextstate_probs[21:25]))
    vyₜ ~ VelCat(softmax(nextstate_probs[26:30]))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@gen function torch_proposal_position(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, occₜ₋₁]
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(
        latent_array, lv_ranges_symbolic), img_dig)
    nextstate_probs ~ nn_torchgen_pos(prevstate_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[21:28]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(pos_to_vel_dist(xₜ, xₜ₋₁))
    vyₜ ~ VelCat(pos_to_vel_dist(yₜ, yₜ₋₁))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@gen function torch_initial_proposal_position(img)
    img_dig = image_digitize(img)
    no_latent_data_and_img_digitized = vcat(lv_pos_no_info(), img_dig)
    nextstate_probs ~ nn_torchgen_pos(no_latent_data_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[21:28]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(uniform(Vels()))
    vyₜ ~ VelCat(uniform(Vels()))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end


@load_generated_functions


train_torch_nn(nn_torchgen_image, torch_proposal_image, @Name(nn_torchgen_full))
#load_torch_nn(nn_torchgen_image, @Name(nn_torchgen_image))

