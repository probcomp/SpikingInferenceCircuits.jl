# module SimpleISParticle

using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits
using .SDCs: IndicatedSpikeCountReal, UnbiasedSpikeCountReal

include("../experiments/sprinkler/implementation_rules.jl")

#####
##### Gen Fn Compilation
#####

@gen (static) function test(input)
    x ~ CPT([[0.5, 0.5], [0.2, 0.8]])(input)
    y ~ CPT([[0.9, 0.1], [0.1, 0.9]])(x)
    z ~ CPT([
        [[0.9, 0.1]] [[0.1, 0.9]];
        [[0.75, 0.25]]  [[0.25, 0.75]]
    ])(x, y)
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

circuit = SIC.ISParticle(test, proposal, 
                     (input = FiniteDomain(2), ),
                     (input = FiniteDomain(2), ))

# Circuits interfaces.
ins = Circuits.inputs(circuit)
outs = Circuits.outputs(circuit)

# println("About to try implementing propose!")
# implemented_assess = implement_deep(circuit.propose, Spiking())
# println("propose implemented.")

# println("About to try implementing assess!")
# implemented_assess = implement_deep(circuit.assess, Spiking())
# println("assess implemented.")

# Implement deep.
implemented = implement_deep(circuit, Spiking())
println("Component implemented.")

include("spiketrain_utils.jl")
#
get_events(impl) = SpikingSimulator.simulate_for_time_and_get_events(
   impl, 500.0; 
   initial_inputs = (
       :propose_args => :input => 1,
       :assess_args => :input => 1,
       :obs => :z => 1
    )
)
#
events = get_events(implemented); nothing
println("Simulation completed.")
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

# end # module
