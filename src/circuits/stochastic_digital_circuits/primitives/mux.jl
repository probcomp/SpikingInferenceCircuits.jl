"""
    Mux <: GenericComponent
    Mux(n_possibilities, out_value)

Multiplexer.  Recieves `n_possibilities` values in the `:values => i` inputs,
and a FiniteDomainValue `:sel`.  Outputs the selected input.
"""
struct Mux <: GenericComponent
    n_possibilities::Int
    out_value::Value
end

Circuits.outputs(m::Mux) = NamedValues(:out => m.out_value)
Circuits.inputs(m::Mux) = NamedValues(
    :values => IndexedValues(m.out_value for _=1:m.n_possibilities),
    :sel => FiniteDomainValue(m.n_possibilities)
)

"""
    OneHotMux(mux, bitmux)

Implementation of `mux` for when the `:sel` FiniteDomainValue(n) is implemented using 
`n` Binary wires with a one-hot encoding.

`bitmux` should be a `BitMux` or implementation of `BitMux` to use.
"""
struct OneHotMux <: GenericComponent
    mux::Mux
    bitmux
end
Circuits.abstract(m::OneHotMux) = m.mux
Circuits.outputs(m::OneHotMux) = outputs(abstract(m))
Circuits.inputs(m::OneHotMux) = NamedValues(
    :values => inputs(abstract(m))[:values],
    :sel => IndexedValues(Binary() for _=1:m.mux.n_possibilities)
)
Circuits.implement(m::OneHotMux, t::Target) =
    let full_in = implement_deep(inputs(m), t),
        full_out = implement_deep(outputs(m), t)
            CompositeComponent(
                full_in, full_out,
                map(v -> bit_muxes(m.bitmux, v), full_in[:values].vals),
                Iterators.flatten(
                    (
                        Input(:values => i_keyname) => CompIn(i_keyname, :value),
                        Input(:sel => i) => CompIn(i_keyname, :sel),
                        CompOut(i_keyname, :out) => Output(i_keyname == i ? :out : :out => keyname)
                    ) for i=1:m.mux.n_possibilities for i_keyname in i_to_keynames(i, full_out[:out])
                )
            )
    end

# Either index at `i => keyname` if there are subkeys, or `i` if there are no subkeys.
i_to_keynames(i, c::CompositeValue) = (i => keyname for keyname in keys_deep(c))
i_to_keynames(i, ::Value) = (i,)

bit_muxes(bm, v::CompositeValue) = ComponentGroup(map(v -> bit_muxes(bm, v), v.vals))
bit_muxes(bm, ::Binary) = bm
bit_muxes(bm, v::Value) = bit_muxes(bm, abstract(v)) # eg. SpikeWire -> Binary

"""
    BitMux <: GenericComponent

Passes through the `:value` Binary input
if the `:sel` Binary input is true.
"""
struct BitMux <: GenericComponent end
Circuits.inputs(m::BitMux) = NamedValues(
        :sel => Binary(), :value => Binary()
    )
Circuits.outputs(m::BitMux) = NamedValues(:out => Binary())

# TODO: we should not call these `Binary`.  We should call it `Line` or `Wire`.