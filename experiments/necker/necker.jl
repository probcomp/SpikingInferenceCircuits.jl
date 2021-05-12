using Gen
using CPTs
using DiscreteIRTransforms
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

includet("implementation_rules.jl")

LabeledCategorical(labels::Vector{T}, probs) where {T} = LabeledCPT{T}([[nothing]], labels, ((_,),) -> probs)

const SIDE_LEN = 1.0
const ROTATION_BOUNDS = collect(-.5:.05:.5)
RotDist = LabeledCategorical(ROTATION_BOUNDS, [1/length(ROTATION_BOUNDS) for _ in ROTATION_BOUNDS])

struct Wrapper{V}
    val::V
end
Wrapper(vals...) = Wrapper((vals...,))
unwrap(w::Wrapper) = w.val

function get_test_vec(w)
    (shape_type, rot_x, rot_z) = unwrap(w)
    vec = ones(50)
    vec[1 + abs(20 * rot_x) |> round |> Int] = 2
    vec[1 + abs(20 * rot_z) |> round |> Int] = 2
    vec[1 + abs(20 * (rot_x + rot_z)) |> round |> Int] = 2
    if shape_type == :cube
        for i=1:50
            vec[i] = vec[i] == 1 ? 2 : 1
        end
    else
        @assert shape_type == :tetrahedron
    end
    return vec
end

const BITFLIPPROB = 0.05
@gen (static) function maybe_flip_bit(value)
    new_value ~ LabeledCPT{Int}([[1, 2]], [1, 2], ((val,),) -> 
        val == 1 ? [1 - BITFLIPPROB, BITFLIPPROB] : [BITFLIPPROB, 1 - BITFLIPPROB]
    )(value)
    return new_value
end

# We have a one-value input, so that when we compile it, we send in a spike to tell it "go"!
@gen (static) function produce_shape_image(input::Nothing)
    shape_type ~ LabeledCategorical([:cube, :tetrahedron], [1/2,1/2])(input)
    rot_x ~ RotDist(input)
    rot_z ~ RotDist(input)
    
    triple = Wrapper(shape_type, rot_x, rot_z)
    pixel_grid::Vector = get_test_vec(triple)
    noisy_pixel_grid ~ Map(maybe_flip_bit)(pixel_grid)
    return noisy_pixel_grid
end

(with_cpts, bijs) = to_indexed_cpts(produce_shape_image, [EnumeratedDomain([nothing])])
@load_generated_functions()

assess_circuit = gen_fn_circuit(with_cpts, (input=FiniteDomain(1),), Assess())

# This takes a while!:
# impl = implement_deep(assess_circuit, Spiking())

