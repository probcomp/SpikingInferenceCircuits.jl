using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

include("../../runs/spiketrain_utils.jl")
include("spiketrain_utils.jl")

includet("implementation_rules.jl")

@compilable @gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.3 : 0.3)
    sprinkler ~ bernoulli(in === nothing ? 0.3 : 0.3)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
    return grasswet
end

# @gen (static) function iswet()
#     raining ~ bernoulli(0.3)
#     sprinkler ~ bernoulli(0.3)
#     grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
#     return grasswet
# end

# Nothing | Bool | Dom([1, 2, 500 1.6]) | Dom(1:0.2:20)






@gen (static) function raining_proposal(raining, sprinkler, grasswet)
    raining ~ bernoulli(raining ? 0.5 : 0.5)
    return raining
end
@compilable raining_proposal(::Dom(1:10), ::1:10, ::Bool)
# raining_proposal = Bundle(...)




@compilable raining_proposal(Bool, Bool, Bool)
@compilable raining_proposal(Dom(1:20), Dom(1:20), Nothing)

# primitive or almost a primitive
MHKernel(bundle, bundle)

@MH_cycle (trace, model_args) |> raining_proposal |> sprinkler_proposal |> raining_proposal |> sprinkler_proposal |> raining_proposal




@gen (static) function sprinkler_proposal(raining, sprinkler, grasswet)
    sprinkler ~ bernoulli(sprinkler ? 0.5 : 0.5)
    return sprinkler
end

# @gen (static) function sprinkler_proposal(raining, sprinkler, grasswet)
#     sprinkler ~ CPT([[0.5 0.5], []])(sprinkler) # 1, 2
#     return sprinkler
# end

trace = initial_trace
while true
    trace = mh(trace, raining_proposal)
    trace = mh(trace, sprinkler_proposal)
end

# @inference function kernel(trace)
#    trace ~ mh(trace, raining_proposal)
#    trace ~ mh(trace, sprinkler_proposal)
# end




Bundle(iswet_cpts, finite_domains...) = @compilable iswet( ::Dom(1: 20) )
MHKernel(b1, b2)

binary = EnumeratedDomain([true, false])
iswet_cpts, _ = to_indexed_cpts(iswet, [EnumeratedDomain([nothing])])
raining_proposal_cpts, _ = to_indexed_cpts(raining_proposal, [binary, binary, binary])
sprinkler_proposal_cpts, _ = to_indexed_cpts(sprinkler_proposal, [binary, binary, binary])

raining_mh_kernel = MHKernel(
    iswet_cpts, (FiniteDomain(1),), # --> gen_fn_circuit
    raining_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2))
)
sprinkler_mh_kernel = MHKernel(iswet_cpts, (in=FiniteDomain(1),), sprinkler_proposal_cpts, (FiniteDomain(2), FiniteDomain(2), FiniteDomain(2)))
println("MH Kernels constructed.")

# rain_impl = implement_deep(raining_mh_kernel, Spiking())
# println("Rain MH Kernel implemented.")

# get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(
#     impl, 1000.;
#     initial_inputs=(
#         :prev_trace => :raining => 1,
#         :prev_trace => :sprinkler => 1,
#         :prev_trace => :grasswet => 1,
#         :model_args => :in => 1
#     )
# )

# events = get_events(rain_impl)
# println("Simulation completed.")

mh_cycle = MH([raining_mh_kernel, sprinkler_mh_kernel, raining_mh_kernel, sprinkler_mh_kernel, raining_mh_kernel, sprinkler_mh_kernel])
###

# raining_  = @gen_fn raining_proposal ([true, false], [true, false], [true, false])

function run_mh(trace, model_arguments)
    while true
        trace = mh(trace, raining_proposal)
        trace = mh(trace, sprinkler_proposal)
    end
end

# function run_smc(x_t, model_args, obs)
#     while true
#         x_t = is(x_t, proposal)
#     end
# end

# println("MH Cycle Constructed.")

# cycle_impl = implement_deep(mh_cycle, Spiking())
# println("MH Cycle implemented.")

# ### Simulate
# using Printf: @sprintf
# function get_log_filter(log_interval)
#     cnt = 0
#     function log_filter(time, compname, event)
#         cnt += 1
#         return (cnt - 1) % log_interval == 0
#     end
#     return log_filter
# end
# function time_log_str(time, compname, event)
#     @sprintf("%.4f", time)
# end

# get_cycle_events(impl, run_time; log=true, log_interval=100) = SpikingSimulator.simulate_for_time_and_get_events(
#     impl, run_time;
#     initial_inputs=(
#         :initial_trace => :raining => 1,
#         :initial_trace => :sprinkler => 1,
#         :initial_trace => :grasswet => 1,
#         :model_args => :in => 1
#     ), log,
#     log_filter=get_log_filter(log_interval),
#     log_str=time_log_str
# )

# events = get_cycle_events(cycle_impl, 1600.); nothing

# # grasswet ss   (assess_new_trace)
# # got 2 spikes in, I think
