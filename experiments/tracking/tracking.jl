using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using DiscreteIRTransforms

include("../sprinkler/implementation_rules.jl")

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
    xₜ   ~ categorical(maybe_one_off(xₜ₋₁, 0.1, XDOMAIN))
    obsₜ ~ categorical(maybe_one_off(xₜ, 0.3, XDOMAIN))
    return obsₜ
end

### Proposal
@gen (static) function step_proposal(xₜ₋₁, obsₜ)
    xₜ ~ categorical(maybe_one_off(obsₜ, 0.2, XDOMAIN))
end

@compilable object_motion_step(::XDOMAIN)
@compilable step_proposal(::XDOMAIN, ::XDOMAIN)

smc_circuit = SMC(object_motion_step, 10, [step_proposal], resample=true)

# @SMC object_motion_step 10 (
#     function smc_step(traces)
#         weighted_traces = is(step_proposal)(traces)
#         traces = resample(weighted_traces)
#     end
# )

# @MH iswet trace -> trace |> mh(sprinkler_proposal) |> mh(raining_proposal)

################
### Circuits ###
################
model_with_cpts, _ = to_indexed_cpts(object_motion_step, [EnumeratedDomain(XDOMAIN)])
proposal_with_cpts, _ = to_indexed_cpts(step_proposal, [EnumeratedDomain(XDOMAIN), EnumeratedDomain(XDOMAIN)])

model_assess_circuit = gen_fn_circuit(model_with_cpts, (xₜ₋₁=FiniteDomain(length(XDOMAIN)),), Assess())
println("Assess circuit constructed.")

proposal_propose_circuit = gen_fn_circuit(proposal_with_cpts, (
        xₜ₋₁=FiniteDomain(length(XDOMAIN)),
        obsₜ=FiniteDomain(length(XDOMAIN))
    ), Assess()
)

println("Propose circuit constructed.")
impl_a = implement_deep(model_assess_circuit, Spiking())
impl_p = implement_deep(proposal_propose_circuit, Spiking())