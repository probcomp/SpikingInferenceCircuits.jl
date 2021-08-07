sdcs = [SpikingInferenceCircuits.SDCs.Mux, 
        SpikingInferenceCircuits.SDCs.ConditionalScore, 
        SpikingInferenceCircuits.SDCs.ConditionalSample, 
        SpikingInferenceCircuits.SDCs.ToAssmts, 
        SpikingInferenceCircuits.SDCs.LookupTable, 
        SpikingInferenceCircuits.SDCs.NonnegativeRealMultiplier, 
        SpikingInferenceCircuits.SDCs.Theta, 
        SpikingInferenceCircuits.SDCs.Step, 
        #SpikingInferenceCircuits.SDCs.CPTSampleScore, 
        SpikingInferenceCircuits.SDCs.MultiInputLookupTable, 
        SpikingInferenceCircuits.SDCs.ValueBlockerPasser]

function inline_node_check(component)
    try
        return any(Circuits.has_abstract_of_type(component, t) for t in sdcs)
    catch e
        return false
    end
end
