##### Step proposal using ANN #####
using ANNDistributions
using Distributions: cdf, Poisson
using ANNDistributions.Flux
import BSON

### Write ANN proposal ###

# This assumes we have already trained the ANN we'll use --
ann = BSON.load("model-checkpoint.bson")[:model]

VelStepDist = ANN_LCPT((Positions(), Vels(), Positions()), Vels(), ann)
@gen (static) function _ann_step_proposal(xₜ₋₁, vₜ₋₁, obs)
    vₜ ~ VelStepDist(xₜ₋₁, vₜ₋₁, obs)
    xₜ ~ Cat(onehot(xₜ₋₁ + vₜ, Positions()))
end

exact_init_proposal = @compile_initial_proposal(_exact_init_proposal, 1)
ann_step_proposal = @compile_step_proposal(_ann_step_proposal, 2, 1)
@load_generated_functions()

### Additional implementation rule needed for ANN: ###
output_maxrate() = 50. # KHz - rate of output from the ANN [e.g. from 100 neuron assemblies with each neuron at 500Hz]
Circuits.implement(a::ANNCPTSample, ::Spiking) =
    ANNDistributions.ConcreteANNCPTSample(
        a; neuron_memory=ΔT()/(length(a.layers) * 2), # make sure that it finishes before ΔT runs out!
        network_memory_per_layer=2 * ΔT()/length(a.layers),
        timer_params=(
            NSPIKES_SYNC_TIMER(),  # N_spikes_timer
            (1, M(), GATE_RATES()...), 0. # timer TI params (maxdelay M gaterates...) | offrate
        ),
        internal_maxrate=10.0, output_maxrate=output_maxrate()
    )
@assert cdf(Poisson(output_maxrate() * MinProb() * ΔT()), RecipPEstDenom()) < 5e-5 "Too high a probability the ANN does not output a score!"
