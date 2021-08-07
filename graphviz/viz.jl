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
Circuits.viz!(inlined; 
              base_path = "graphviz/velwalk1d")
inlined, names, state = Circuits.inline(impl;
                                        treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_velwalk1d")

#X_init = 3
#Y_init = 0
#Z_init = 5
#includet("3dline.jl")
#inlined, names, state = Circuits.inline(impl)
#Circuits.viz!(inlined; 
#              base_path = "graphviz/3dline")
#inlined, names, state = Circuits.inline(impl;
#                    treat_as_primitive = inline_node_check)
#Circuits.viz!(state; 
#              base_path = "graphviz/sdc_3dline")

includet("occlusion_tracking.jl")
inlined, names, state = Circuits.inline(impl)
Circuits.viz!(inlined; 
              base_path = "graphviz/occlusion")
inlined, names, state = Circuits.inline(impl;
                                        treat_as_primitive = inline_node_check)
Circuits.viz!(state; 
              base_path = "graphviz/sdc_occlusion")
