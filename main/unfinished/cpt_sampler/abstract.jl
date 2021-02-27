struct AbstractCPTSampler{C} <: GenericComponent
    cpt::C
    AbstractCPTSampler(cpt::C) where {C <: CPT} = new{C}(cpt)
end
# performance TODO: pass in iterator, not tuple?
inputs(s::AbstractCPTSampler) = CompositeValue(Tuple(FiniteDomainValue(n) for n in s.cpt.input_domain_sizes))
outputs(s::AbstractCPTSampler) = CompositeValue((FiniteDomainValue(s.cpt.output_domain_size),))