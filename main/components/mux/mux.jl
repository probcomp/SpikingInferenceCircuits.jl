struct BitMux <: GenericComponent end
Circuits.inputs(m::BitMux) = NamedValues(
        :sel => Binary(), :value => Binary()
    )
Circuits.outputs(m::BitMux) = NamedValues(:out => Binary())

struct Mux <: GenericComponent
    n_possibilities::Int
    out_value::CompositeValue
end
Circuits.outputs(m::Mux) = m.out_value
Circuits.inputs(m::Mux) = NamedValues(
    :values => IndexedValues(m.out_value for _=1:m.n_possibilities),
    :sel => IndexedValues(Binary() for _=1:m.n_possibilities)
)

# For any target, we can implement a Mux using the `BitMux` for the 
Circuits.implement(m::Mux, t::Target) =
    if can_implement(BitMux(), t) && compiles_to_binary(outputs(m))
        _bit_mux_implementation(m)
    else
        error("Mux not implemented for $t.  If $t implemented BitMux(), we could implement in terms of this.")
    end

_bit_mux_implementation(m::Mux) =
    let full_in = implement_deep(inputs(m), t),
        full_out = implement_deep(inputs(m), t),
        n_passthrough = length_deep(full_out)
            CompositeComponent(
                full_in, full_out,
                (BitMux() for _ in keys_deep(full_in)),
                Iterators.flatten((
                    (
                        Input(:sel => keyname) => CompIn((i-1)*n_passthrough + k, :sel)
                        for i=1:m.n_possibilities,
                            (k, keyname) in enumerate(keys_deep(full_out))
                    ),
                    (
                        Input(:value => keyname) => CompIn((i-1)*n_passthrough + k, :value)
                        for i=1:m.n_possibilities,
                            (k, keyname) in enumerate(keys_deep(full_out))
                    ),
                    (
                        CompIn((i-1)*n_passthrough + k, :out) => Output(keyname)
                        for i=1:m.n_possibilities,
                            (k, keyname) in enumerate(keys_deep(full_out))
                    )
                ))
            )
    end