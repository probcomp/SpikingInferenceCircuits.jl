using Circuits
using CircuitViz
import Circuits:add_node!

using SpikingInferenceCircuits
using ProbEstimates

includet("velwalk1d.jl")

using SpikingInferenceCircuits.SDCs: Mux, 
                                     ConditionalScore,
                                     ConditionalSample, 
                                     ToAssmts,
                                     LookupTable,
                                     NonnegativeRealMultiplier,
                                     Theta,
                                     Step,
                                     MultiInputLookupTable,
                                     ValueBlockerPasser

sdcs = [Mux, 
        ConditionalScore, 
        ConditionalSample, 
        ToAssmts, 
        LookupTable, 
        NonnegativeRealMultiplier, 
        Theta, 
        Step, 
        #CPTSampleScore, 
        MultiInputLookupTable, 
        ValueBlockerPasser]

function add_node!(g, name, cc::CompositeComponent)
    i = findfirst([has_abstract_of_type(cc, t) for t in sdcs])
    type = sdcs[i]
    node_attrs = Dict{Symbol, Any}(:shape => "square",
                                   :label => repr(type))
    CircuitViz.add_vertex!(g, node_attrs)

end

function inline_node_check(component)
    try
        return any(Circuits.has_abstract_of_type(component, t) for t in sdcs)
    catch e
        return false
    end
end

inlined, names, state = Circuits.inline(impl)
ks_vel = keys(state.internals)

# Full.
Circuits.viz!(inlined; 
              base_path = "graphviz/velwalk1d")
inlined, names, state = Circuits.inline(impl;
                                        treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_velwalk1d")

# Sub-circuits.
_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :propose])
Circuits.viz!(state; 
              base_path = "graphviz/velwalk1d_propose")
_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :propose]; treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_velwalk1d_propose")

_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :assess_latents])
Circuits.viz!(state; 
              base_path = "graphviz/velwalk1d_assess_latents")
_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :assess_latents]; treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_velwalk1d_assess_latents")

_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :assess_obs])
Circuits.viz!(state; 
              base_path = "graphviz/velwalk1d_assess_obs")
_, _, state = Circuits.inline(impl[:subsequent_steps => :smcstep => :particles => 1 => :assess_obs]; treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_velwalk1d_assess_obs")
