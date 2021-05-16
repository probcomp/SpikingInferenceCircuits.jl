using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

includet("../logging_utils.jl")
includet("implementation_rules.jl")
includet("spiketrain_utils.jl")

# UTILS:
# onehot vector for `x` with length `length(dom)`,
# with `x` truncated to domain
onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]

# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)

### Model
XDOMAIN = 1:10
@gen (static) function object_motion_step(xₜ₋₁)
    xₜ   ~ categorical(maybe_one_off(xₜ₋₁, 0.8, XDOMAIN))
    obsₜ ~ categorical(maybe_one_off(xₜ, 0.5, XDOMAIN))
    return obsₜ
end

### Proposal
@gen (static) function step_proposal(xₜ₋₁, obsₜ)
    xₜ ~ categorical(maybe_one_off(obsₜ, 0.5, XDOMAIN))
end
@load_generated_functions()

################
### Circuits ###
################
model_with_cpts, _ = to_indexed_cpts(object_motion_step, [EnumeratedDomain(XDOMAIN)])
proposal_with_cpts, _ = to_indexed_cpts(step_proposal, [EnumeratedDomain(XDOMAIN), EnumeratedDomain(XDOMAIN)])

model_in_domains = (xₜ₋₁=FiniteDomain(length(XDOMAIN)),)
proposal_in_domains = (
            xₜ₋₁=FiniteDomain(length(XDOMAIN)),
            obsₜ=FiniteDomain(length(XDOMAIN))
        )


# model_assess_circuit = gen_fn_circuit(model_with_cpts, model_in_domains, Assess())
# println("Assess circuit constructed.")
# proposal_propose_circuit = gen_fn_circuit(proposal_with_cpts, proposal_in_domains, Assess())

# println("Propose circuit constructed.")
# impl_a = implement_deep(model_assess_circuit, Spiking())
# impl_p = implement_deep(proposal_propose_circuit, Spiking())

NPARTICLES() = 4
smc_circuit = SIC.SMC(NPARTICLES(), model_with_cpts, proposal_with_cpts, model_in_domains, proposal_in_domains)
println("SMC circuit constructed.")

smc_impl = Circuits.memoized_implement_deep(smc_circuit, Spiking())

get_events(impl, runtime, inputs; log_interval=400) =
    SpikingSimulator.simulate_for_time_and_get_events(
        impl, runtime;
        inputs,
        log=true,
        log_filter=get_log_filter(log_interval),
        log_str=time_log_str
    )

function get_gt_traces_and_events(
    impl, runtime;
    inter_obs_interval=400., log_interval=400,
    initial_x=5
)
    (gt_traces, inputs) = get_gt_traces_and_inputs(runtime, inter_obs_interval, initial_x)
    
    return (
        gt_traces,
        get_events(impl, runtime, inputs; log_interval)
    )
end

function get_gt_traces_and_inputs(runtime, inter_obs_interval, initial_x)    
    initial_tr = simulate(object_motion_step, (initial_x,))
    gt_traces = [initial_tr]
    inputs = Tuple{Float64, Tuple}[
        (0., (
            (
                :initial_latents => i => :xₜ₋₁ => initial_x
                for i=1:NPARTICLES()
            )...,
            :obs => :obsₜ => initial_tr[:obsₜ])
        )
    ]
    current_x = initial_tr[:xₜ]
    t = 0.
    while t < runtime
        t += inter_obs_interval
        tr = simulate(object_motion_step, (current_x,))
   
        push!(gt_traces, tr)
        push!(inputs, (t, (:obs => :obsₜ => tr[:obsₜ],)))
        current_x = tr[:xₜ]
    end

    return (gt_traces, inputs)
end

trs, ins = get_gt_traces_and_inputs(2000, 390, 5)
display([(tr[:xₜ], tr[:obsₜ]) for tr in trs])
events = get_events(smc_impl, 2000, ins)