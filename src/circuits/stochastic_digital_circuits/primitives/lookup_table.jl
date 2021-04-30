# TODO: I'm not sure we actually need or want this component.

"""
    LookupTable <: GenericComponent

A function from a FiniteDomainValue to a FiniteDomainValue.
"""
struct LookupTable <: GenericComponent
    input_domain_size::Int
    output_domain_size::Int
    f::Function    
end
Circuits.inputs(lt::LookupTable) = NamedValues(:in => FiniteDomainValue(lt.input_domain_size))
Circuits.outputs(lt::LookupTable) = NamedValues(:out => FiniteDomainValue(lt.output_domain_size))