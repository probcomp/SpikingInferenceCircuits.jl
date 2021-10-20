using PyCall
using GenPyTorch
using Statistics
using Gen
import Flux: softmax


include("model.jl")
include("groundtruth_rendering.jl")
include("visualize.jl")
include("obs_aux_proposal.jl")
include("prior_proposal.jl")
include("nearly_locally_optimal_proposal.jl")
include("run_utils.jl")
include("obs_aux_proposal.jl")
include("visualize.jl")
include("ann_utils.jl")


torch = pyimport("torch")
nn = torch.nn
F = nn.functional


tdr, td, t_im, gt_trs = generate_training_data(1, 15, digitize_trace)
input_dp = convert(Vector{Float64}, td[1][1])

partition_nn_output(y) = [softmax(y[i1+1:i2]) for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]



@pydef mutable struct SingleHidden <: nn.Module
    function __init__(self, input_dp)
        # Note the use of pybuiltin(:super): built in Python functions
        # like `super` or `str` or `slice` are all accessed using
        # `pybuiltin`.
        pybuiltin(:super)(SingleHidden, self).__init__()
        self.dense1 = nn.Linear(length(input_dp),
                                Int(round(mean([length(input_dp), sum([length(l) for l in lv_ranges])]))))
        self.dense2 = nn.Linear(Int(round(mean([length(input_dp), sum([length(l) for l in lv_ranges])]))),
                                sum([length(l) for l in lv_ranges]))
    end

    function forward(self, x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
#        output = vcat([F.softmax(y) for y in partition_nn_output(x)])
        # for some reason can't use ... syntax here
        #        return output[1], output[2], output[3], output[4], output[5]
#        println(typeof(x))
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

nn_mod = SingleHidden(input_dp)

nn_torchgen = TorchGenerativeFunction(nn_mod, [TorchArg(true, torch.float)], 1)


@gen (static) function pytorch_proposal(occₜ₋₁, xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, img)
    latent_array = [xₜ₋₁, yₜ₋₁, vxₜ₋₁, vyₜ₋₁, occₜ₋₁]
    img_dig = image_digitize(img)
    prevstate_and_img_digitized = vcat(lv_to_onehot_array(latent_array, lv_ranges), img_dig)
    nextstate_array ~ nn_torchgen(prevstate_and_img_digitized)
    nextstate_probs = partition_nn_output(nextstate_array)
    occₜ ~ Cat(nextstate_probs[end])
    xₜ ~ Cat(nextstate_probs[1])
    yₜ ~ Cat(nextstate_probs[2])
    vxₜ ~ VelCat(nextstate_probs[3])
    vyₜ ~ VelCat(nextstate_probs[4])
    return (occₜ, xₜ, yₜ, vxₜ, vyₜ)
end

@load_generated_functions


# in example, measurements is the input to the proposal.
# construct this by sampling a one step model each time and properly assigning
# the previous states and image to the model. 

function groundtruth_generator()
    
    # since these names are used in the global scope, explicitly declare it
    # local to avoid overwriting the global variable
    # obtain an execution of the model where planning succeeded
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


update = Gen.ParamUpdate(Gen.ADAM(0.001, 0.9, 0.999, 1e-8), 
                         nn_torchgen => collect(get_params(nn_torchgen)))

Gen.train!(pytorch_proposal, groundtruth_generator, update,
    num_epoch=10, epoch_size=100, num_minibatch=100, minibatch_size=100,
    evaluation_size=10, verbose=true);
