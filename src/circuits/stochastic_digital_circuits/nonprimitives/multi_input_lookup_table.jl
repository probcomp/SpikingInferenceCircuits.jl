struct MultiInputLookupTable{n} <: GenericComponent
    input_domain_sizes::NTuple{n, Int}
    output_domain_size::Int
    f::Function
end
Circuits.inputs(lt::MultiInputLookupTable) = NamedValues(
    :inputs => IndexedValues(
            FiniteDomainValue(s) for s in lt.input_domain_sizes
        )
    )
Circuits.outputs(lt::MultiInputLookupTable) = NamedValues(:out => FiniteDomainValue(lt.output_domain_size))

Circuits.implement(lt::MultiInputLookupTable, ::Target) =
    CompositeComponent(
        inputs(lt), outputs(lt), (
            to_assmts=ToAssmts(lt.input_domain_sizes),
            lt=LookupTable(prod(input_domain_sizes), lt.output_domain_size, lt.f)
        ),
        (
            (
                Input(i) => CompIn(:to_assmts, i)
                for i=1:length(lt.input_domain_sizes)
            )...,
            CompOut(:to_assmts, :out) => CompIn(:lt, :in),
            CompOut(:lt, :out) => Output(:out)
        )
    )