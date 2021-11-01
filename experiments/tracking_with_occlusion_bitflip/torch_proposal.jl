using PyCall
using GenPyTorch
using Statistics
using Gen
import Flux: softmax
import Base: zero

include("model.jl")
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

tdr, td, t_im, gt_trs = generate_training_data(1, 15, digitize_trace)
input_dp = convert(Vector{Float64}, td[1][1])
maxlen_lv_range = maximum(map(f-> length(f), lv_ranges))
partition_nn_output(y) = [softmax(y[i1+1:i2]) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]
lv_no_info() = zeros(sum([length(l) for l in lv_ranges]))


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


#@gen (static) function 

@gen function torch_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate_probs ~ nn_torchgen(prevstate_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[31:38]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(softmax(nextstate_probs[21:25]))
    vyₜ ~ VelCat(softmax(nextstate_probs[26:30]))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@gen function torch_initial_proposal(img)
    img_dig = image_digitize(img)
    no_latent_data_and_img_digitized = vcat(lv_no_info(), img_dig)
    nextstate_probs = nn_torchgen(no_latent_data_and_img_digitized)
    occₜ ~ Cat(softmax(nextstate_probs[31:38]))
    xₜ ~ Cat(softmax(nextstate_probs[1:10]))
    yₜ ~ Cat(softmax(nextstate_probs[11:20]))
    vxₜ ~ VelCat(softmax(nextstate_probs[21:25]))
    vyₜ ~ VelCat(softmax(nextstate_probs[26:30]))
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

nn_mod = SingleHidden(input_dp)
nn_torchgen = TorchGenerativeFunction(nn_mod, [TorchArg(true, torch.float)], 1)


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
    #latents choicemap is the choicemap itself -- might be helpful. 
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


function train_torch_nn()
    parameter_update = Gen.ParamUpdate(Gen.ADAM(0.001, 0.9, 0.999, 1e-8), 
                                       nn_torchgen => collect(get_params(nn_torchgen)))
    Gen.train!(torch_proposal, groundtruth_generator, parameter_update,
               num_epoch=10, epoch_size=100, num_minibatch=100, minibatch_size=100,
               evaluation_size=10, verbose=true);
end

