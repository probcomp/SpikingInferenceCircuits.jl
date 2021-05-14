module SimpleISParticle

using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using .SDCs: IndicatedSpikeCountReal, UnbiasedSpikeCountReal

### Implementation Rules ###

Circuits.implement(ta::SIC.SDCs.ToAssmts, ::Spiking) =
SDCs.PulseToAssmts(
                   ta, PulseIR.PoissonThresholdedIndicator,
                   # ΔT, max_delay, M, R
                   (500, 0.5, 50, 40)
                   # Note that the R needs to be high since getting spikes while off is catastrophic.
                   # TODO: Design things so this is not catastrophic (or can't happen at
                   # realistic rates)!
                  )

K = 10
ONRATE = 0.2
Circuits.implement(cs::SIC.SDCs.ConditionalSample, ::Spiking) =
SDCs.PoissonPulseConditionalSample(
                                   (cs, K, ONRATE,
                                    500, # ΔT
                                    0.2, # max_delay
                                    1000, # M (num spikes to override offs/ons)
                                    50, # max delay before sample is emitted
                                    0.1 # intersample hold
                                   ),
                                   10^(-10), 12
                                  )
Circuits.implement(cs::SIC.SDCs.ConditionalScore, ::Spiking) =
SDCs.PoissonPulseConditionalScore((cs, K, ONRATE, 500, 0.2, 1000), 10^(-10), 12)

Circuits.implement(lt::SIC.SDCs.LookupTable, ::Spiking) =
SIC.SDCs.OneHotLookupTable(lt)

Circuits.implement(::Binary, ::Spiking) = SpikeWire()

to_spiking_real(::SDCs.SingleNonnegativeReal) = 
IndicatedSpikeCountReal(UnbiasedSpikeCountReal(K))
to_spiking_real(v::SDCs.ProductNonnegativeReal) =
SDCs.ProductNonnegativeReal(map(to_spiking_real, v.factors))
to_spiking_real(v::SDCs.NonnegativeReal) = to_spiking_real(implement(v, Spiking()))

Circuits.implement(r::SDCs.SingleNonnegativeReal, ::Spiking) = to_spiking_real(r)
Circuits.implement(r::SDCs.ProductNonnegativeReal, ::Spiking) = to_spiking_real(r)

Circuits.implement(m::SDCs.NonnegativeRealMultiplier, ::Spiking) = 
SDCs.PulseNonnegativeRealMultiplier(
                                    map(to_spiking_real, m.inputs),
                                    (indenoms, outdenom) -> PulseIR.PoissonSpikeCountMultiplier(
                                                                                                indenoms, outdenom,
                                                                                                10, 50., 300., 0.5,
                                                                                                ((500, 12), 0.), #(ti_params, offrate)
                                                                                                (500, 12)
                                                                                               ),
                                    K,
                                    threshold -> begin
                                        println("thresh: $threshold")
                                        PulseIR.PoissonThresholdedIndicator(threshold, 500, 0.5, 50, 40)
                                    end
                                   )

#####
##### Gen Fn Compilation
#####

@gen (static) function test(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    z ~ CPT([[[0.9, 0.1]] 
             [[0.1, 0.9]];
             [[0.75, 0.25]] 
             [[0.25, 0.75]]])(x, y)
    return z
end

@gen (static) function proposal(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    #z ~ CPT([[[0.9, 0.1]] 
    #         [[0.1, 0.9]];
    #         [[0.75, 0.25]] 
    #         [[0.25, 0.75]]])(x, y)
    return y
end

#####
##### IS Particle
#####

# Circuit with prior as proposal.
println("IS Particle TEST")
println()

circuit = ISParticle(test, proposal, 
                     (input = FiniteDomain(2), ),
                     (input = FiniteDomain(2), ))

# Circuits interfaces.
ins = Circuits.inputs(circuit)
outs = Circuits.outputs(circuit)

# Implement deep.
implemented = implement_deep(circuit, Spiking())
println("Component implemented.")

#include("spiketrain_utils.jl")
#
#get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(
#    impl, 500.0; 
#    initial_inputs = (:inputs => :input => 1,)
#)
#
#events = get_events(implemented)
#println("Simulation completed.")
#println(out_st_dict(events))
# draw_fig(events)  

#println()
#println()
#println("ASSESS TEST")
#println()
#circuit2 = gen_fn_circuit(test, (input=FiniteDomain(2),), Assess())
#
#implemented2 = implement_deep(circuit2, Spiking())
#
#println("Component implemented.")
#
#get_events2(impl) = SpikingSimulator.simulate_for_time_and_get_events(
#    impl, 500.0; initial_inputs=(
#        :inputs => :input => 1,
#        :obs => :x => 2,
#        :obs => :y => 2,
#        :obs => :z => 1
#    )
#)
#
#events2 = get_events2(implemented2)
#println("Simulation completed.")
#println(out_st_dict(events2))

end # module
