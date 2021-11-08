using Circuits, SpikingInferenceCircuits, SpikingCircuits
const SIC = SpikingInferenceCircuits

include("../../model/model.jl")
include("../../groundtruth_rendering.jl")

includet("../../../utils/default_implementation_rules.jl")
println("Implementation rules loaded.")

num_neurons(::PrimitiveComponent) = 1
num_neurons(c::CompositeComponent) = reduce(+, map(num_neurons, c.subcomponents); init=0.)

latent_domains()     = (
    occₜ = positions(OccluderLength()),
    xₜ   = positions(SquareSideLength()),
    yₜ   = positions(SquareSideLength()),
    vxₜ  = Vels(),
    vyₜ  = Vels()
)
xyocc_domains() = (latent_domains()[:xₜ], latent_domains()[:yₜ], latent_domains()[:occₜ])

ImgVecIndices() = 1:(3^(ImageSideLength()^2))
img_of_color_indices(x, y, occ) = [pix == Empty() ? 0 : pix == Object() ? 1 : 2 for pix in image_determ(occ, x, y)]
img_to_idx(img) = 1 + sum(val * 3^i for (i, val) in enumerate(img))
img_pvec(x, y, occ) = onehot(img_to_idx(img_of_color_indices(x, y, occ)), ImgVecIndices())
@gen (static) function cpt_gen_fn(x, y, occ)
    img ~ Cat(img_pvec(x, y, occ))
    return img
end

function compile_cpt()
    circuit = gen_fn_circuit(
        GenFnWithInputDomains(cpt_gen_fn, xyocc_domains()), Assess()
    )
    impl = implement_deep(circuit, Spiking())
    return impl
end

function compile_assess()
    circuit = gen_fn_circuit(
        SIC.replace_return_node(
            GenFnWithInputDomains(obs_model, latent_domains())
        ),
        Assess()
    )
    impl = implement_deep(circuit, Spiking())
    return impl
end
@load_generated_functions()

cpt_sizes = []
assess_sizes = []
for size in (1, 2, 3)
    ImageSideLength() = size
    OccluderLength() = min(size, 2)

    cpt_impl = compile_cpt()
    cnt = num_neurons(cpt_impl)
    println("$cnt neurons for CPT with size = $size")
    push!(cpt_sizes, cnt)
end

for size in 1:6
    ImageSideLength() = size
    OccluderLength() = min(size, 2)

    assess_impl = compile_assess()
    cnt = num_neurons(assess_impl)
    println("$cnt neurons for ASSESS with size = $size")
    push!(assess_sizes, cnt)
end

println("CPT Sizes: $cpt_sizes")
println("ASSESS Sizes: $assess_sizes")