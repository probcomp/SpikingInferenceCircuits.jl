using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

include("../../runs/spiketrain_utils.jl")
include("spiketrain_utils.jl")

includet("implementation_rules.jl")

# util for debugging:
pairlast(p::Pair) = pairlast(p.second)
pairlast(v) = v

@gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.3 : 0.3)
    sprinkler ~ bernoulli(in === nothing ? 0.3 : 0.3)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
    return grasswet
end

@gen (static) function raining_proposal(raining, sprinkler, grasswet)
    raining ~ bernoulli(raining ? 0.5 : 0.5)
    return raining
end

@gen (static) function sprinkler_proposal(raining, sprinkler, grasswet)
    sprinkler ~ bernoulli(sprinkler ? 0.5 : 0.5)
    return sprinkler
end

@gen (static) function smart_block_proposal(raining, sprinkler, grasswet)
    raining ~ bernoulli(grasswet ? 0.55 : 0.2)
    sprinkler ~ bernoulli(grasswet && !raining ? 0.8 : 0.3)
    return sprinkler
end

binary = EnumeratedDomain([true, false])
iswet_cpts, _ = to_indexed_cpts(iswet, [EnumeratedDomain([nothing])])
raining_proposal_cpts, _ = to_indexed_cpts(raining_proposal, [binary, binary, binary])
sprinkler_proposal_cpts, _ = to_indexed_cpts(sprinkler_proposal, [binary, binary, binary])
block_proposal_cpts, _ = to_indexed_cpts(smart_block_proposal, [binary, binary, binary])

raining_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), raining_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
sprinkler_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), sprinkler_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
block_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), block_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
println("MH Kernels constructed.")

# mh_cycle = MH([raining_mh_kernel, sprinkler_mh_kernel, raining_mh_kernel, sprinkler_mh_kernel, raining_mh_kernel, sprinkler_mh_kernel])
# println("MH Cycle Constructed.")

mh_cycle = MH([block_mh_kernel for _=1:6])
println("MH Cycle Constructed.")

cycle_impl = Circuits.memoized_implement_deep(mh_cycle, Spiking())
println("MH Cycle implemented.")

### Simulate
using Printf: @sprintf
function get_log_filter(log_interval)
    cnt = 0
    function log_filter(time, compname, event)
        cnt += 1
        return (cnt - 1) % log_interval == 0
    end
    return log_filter
end
function time_log_str(time, compname, event)
    @sprintf("%.4f", time)
end

get_cycle_events(impl, run_time; log=true, log_interval=400) = SpikingSimulator.simulate_for_time_and_get_events(
    impl, run_time;
    initial_inputs=(
        :initial_trace => :raining => 1,
        :initial_trace => :sprinkler => 1,
        :initial_trace => :grasswet => 1,
        :model_args => :in => 1
    ), log,
    log_filter=get_log_filter(log_interval),
    log_str=time_log_str
)

events = get_cycle_events(cycle_impl, 10_000.); nothing

# grasswet ss   (assess_new_trace)
# got 2 spikes in, I think
