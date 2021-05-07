"""
    LookupTable <: GenericComponent

A function from a `{1, …, input_domain_size}` to a `{1, …, output_domain_size}`.
"""
struct LookupTable <: GenericComponent
    input_domain_size::Int
    output_domain_size::Int
    f::Function    
end
Circuits.inputs(lt::LookupTable) = NamedValues(:in => FiniteDomainValue(lt.input_domain_size))
Circuits.outputs(lt::LookupTable) = NamedValues(:out => FiniteDomainValue(lt.output_domain_size))

"""
A lookup table where finite-domain values are encoded using one-hot
encoding on `Binary` values.
"""
struct OneHotLookupTable <: GenericComponent
    lt::LookupTable
end
Circuits.abstract(lt::OneHotLookupTable) = lt.lt
Circuits.inputs(lt::OneHotLookupTable) = NamedValues(:in => 
    CompositeValue(Tuple(Binary() for _=1:lt.lt.input_domain_size), FiniteDomainValue(lt.lt.input_domain_size))
)
Circuits.outputs(lt::OneHotLookupTable) = NamedValues(:out => 
CompositeValue(Tuple(Binary() for _=1:lt.lt.output_domain_size), FiniteDomainValue(lt.lt.output_domain_size))
)
Circuits.implement(lt::OneHotLookupTable, ::Target) = CompositeComponent(
        inputs(lt), outputs(lt), (), (
            Input(:in => inval) => Output(:out => lt.lt.f(inval))
            for inval=1:lt.lt.input_domain_size
        ), lt
    )