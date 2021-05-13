using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

includet("implementation_rules.jl")

@gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.2 : 0.2)
    sprinkler ~ bernoulli(in === nothing ? 0.2 : 0.2)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
end

@gen (static) function raining_proposal(raining, sprinkler, grasswet)
    raining ~ bernoulli(raining ? 0.2 : 0.8)
end

@gen (static) function sprinkler_proposal(raining, sprinkler, grasswet)
    sprinkler ~ bernoulli(sprinkler ? 0.2 : 0.8)
end

function inference_cycle(tr)
    tr = mh(tr, raining_proposal)
    tr = mh(tr, sprinkler_proposal)
    return tr
end

binary = EnumeratedDomain([true, false])
iswet_cpts, _ = to_indexed_cpts(iswet, [EnumeratedDomain([nothing])])
raining_proposal_cpts, _ = to_indexed_cpts(raining_proposal, [binary, binary, binary])
sprinkler_proposal_cpts, _ = to_indexed_cpts(sprinkler_proposal, [binary, binary, binary])

raining_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), raining_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
sprinkler_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), sprinkler_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
println("MH Kernels constructed.")

rain_impl = implement_deep(raining_mh_kernel, Spiking())
println("Rain MH Kernel implemented.")

include("../../runs/spiketrain_utils.jl")

get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(
    impl, 1000.;
    initial_inputs=(
        :prev_trace => :raining => 1,
        :prev_trace => :sprinkler => 1,
        :prev_trace => :grasswet => 1,
        :model_args => :in => 1
    )
)

events = get_events(rain_impl)
println("Simulation completed.")

