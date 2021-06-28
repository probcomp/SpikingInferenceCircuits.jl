using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using Circuits, SpikingCircuits
includet("implementation_rules.jl")
println("Implementation rules loaded.")

include("compilable_model.jl")

latent_domains() = (Positions(), Vels(), Positions(), Vels())
obs_domains() = (Positions(), Positions())
latent_obs_domains() = (latent_domains()..., obs_domains()...)
NPARTICLES() = 2

# rsmcstep = RecurrentSMCStep(
#     SMCStep(
#         GenFnWithInputDomains(step_model, latent_domains()),
#         GenFnWithInputDomains(obs_model, latent_domains()),
#         GenFnWithInputDomains(step_proposal, latent_obs_domains()),
#         [:xₜ, :vxₜ, :yₜ, :vyₜ],
#         [:obsx, :obsy],
#         NPARTICLES()
#     ),
#     [:xₜ, :vxₜ, :yₜ, :vyₜ]
# )
# println("SMC Circuit Constructed.")

# impl = Circuits.memoized_implement_deep(rsmcstep, Spiking());
# println("Circuit implemented deeply.")

# include("../simulation_utils.jl")

# inputs = get_smc_circuit_inputs(
#     10000, 1000,
#     (
#         xₜ₋₁ = 2, yₜ₋₁ = 18,
#         vxₜ₋₁ = 2 + (1 - first(Vels())),
#         vyₜ₋₁ = -2 + (1 - first(Vels()))
#     ), [
#         (obsx = x, obsy = y)
#         for (x, y) in [
#             (2, 16), (6, 14), (9, 11),
#             (11, 10), (11, 7), (12, 6),
#             (18, 6), (18, 4), (19, 1),
#             (19, 1), (19, 1), (19, 1)
#         ]
#     ],
#     NPARTICLES()
# )
# println("Constructed input spike sequence.")

# using Serialization
# function save_evts(evts, run_idx)
#     try
<<<<<<< HEAD
#         serialize("run$(run_idx)_20210628_events.jls", evts)
=======
#         serialize("run$(run_idx)_20210624_events.jls", evts)
>>>>>>> 446bf64529fadd83065f35839e1ecc7f938d727c
#     catch e
#         @error("Error saving events for run $run_idx.")
#     end
# end

# function do_run(i)
<<<<<<< HEAD
#     events = simulate_and_get_events(impl, 999., inputs)
=======
#     events = simulate_and_get_events(impl, 10000, inputs)
>>>>>>> 446bf64529fadd83065f35839e1ecc7f938d727c
#     save_evts(events, i)
#     return events
# end

<<<<<<< HEAD
# println("Have access to $(Threads.nthreads()) threads")
# evts = Any[nothing for _=1:4]
# Threads.@threads for i=1:4
#     println("Beginning $(i)th simulation.")
#     evts[i] = do_run(i)
=======
# evts = Any[]
# for i=1:4
#     println("Beginning $(i)th simulation.")
#     push!(evts, do_run(i))
>>>>>>> 446bf64529fadd83065f35839e1ecc7f938d727c
#     println("Completed $(i)th simulation.")
# end