using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
include("compilable_model.jl")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 4

rsmcstep = RecurrentSMCStep(
    SMCStep(
        GenFnWithInputDomains(step_model, latent_domains()),
        GenFnWithInputDomains(obs_model, latent_domains()),
        GenFnWithInputDomains(step_proposal, latent_obs_domains()),
        [:xₜ, :vxₜ, :yₜ, :vyₜ],
        [:obsx, :obsy],
        NPARTICLES()
    ),
    [:xₜ, :vxₜ, :yₜ, :vyₜ]
)
println("SMC Circuit Constructed.")

includet("implementation_rules.jl")
println("Implementation rules loaded.")
impl = Circuits.memoized_implement_deep(rsmcstep, Spiking());
println("Circuit implemented deeply.")

include("../simulation_utils.jl")

inputs = get_smc_circuit_inputs(
    10000, 1000,
    (
        xₜ₋₁ = 2, yₜ₋₁ = 18,
        vxₜ₋₁ = 2 + (1 - first(Vels())),
        vyₜ₋₁ = -2 + (1 - first(Vels()))
    ), [
        (obsx = x, obsy = y)
        for (x, y) in [
            (2, 16), (6, 14), (9, 11),
            (11, 10), (11, 7), (12, 6),
            (18, 6), (18, 4), (19, 1),
            (19, 1), (19, 1), (19, 1)
        ]
    ],
    NPARTICLES()
)
println("Constructed input spike sequence.")

using Serialization
function save_evts(evts, run_idx)
    try
        serialize("saves/run$(run_idx)_20210623_events.jls", evts)
    catch e
        @error("Error saving events for run $run_idx.")
    end
    
end

function do_run(i)
    events = simulate_and_get_events(impl, 10000, inputs)
    save_evts(events, i)
    return events
end

evts = Any[]
for i=1:4
    println("Beginning $(i)th simulation.")
    push!(evts, do_run(i))
    println("Completed $(i)th simulation.")
end