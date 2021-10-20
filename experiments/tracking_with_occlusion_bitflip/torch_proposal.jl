using PyCall
using GenPyTorch
using Statistics

include("obs_aux_proposal.jl")
include("model.jl")
include("groundtruth_rendering.jl")
include("prior_proposal.jl")
include("visualize.jl")
include("ann_utils.jl")


torch = pyimport("torch")
nn = torch.nn
F = nn.functional


tdr, td, t_im, gt_trs = generate_training_data(1, 15, digitize_trace)
input_dp = convert(Vector{Float64}, td[1][1])

function partition_nn_output(y)
    vec_y = y[:values]
    [vec_y[i1+1:i2] for (i1, i2) in sliding_window(vcat(0, my_cumsum([length(r) for r in lv_ranges])))]
end


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

model_gf = TorchGenerativeFunction(nn_mod, [TorchArg(true, torch.float)], 1)
