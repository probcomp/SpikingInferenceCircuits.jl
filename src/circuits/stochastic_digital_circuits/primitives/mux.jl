"""
    Mux <: GenericComponent
    Mux(n_possibilities, out_value)

Multiplexer.  Recieves `n_possibilities` values in the `:values => i` inputs,
and a FiniteDomainValue `:sel`.  Outputs the selected input.
"""
struct Mux <: GenericComponent
    n_possibilities::Int
    out_value::CompositeValue
end

Circuits.outputs(m::Mux) = m.out_value
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
    :values => IndexedValues(m.out_value for _=1:m.n_possibilities),
    :sel => IndexedValues((Binary() for _=1:m.n_possibilities), FiniteDomainValue(m.n_possibilities))
)
Circuits.implement(m::OneHotMux, ::Target) =
    let full_in = implement_deep(inputs(m), t),
        full_out = implement_deep(outputs(m), t)
            CompositeComponent(
                full_in, full_out,
                map(v -> bit_muxes(m.bitmux, v), full_in[:values].vals),
                Iterators.flatten(
                    (
                        Input(:values => i => keyname) => CompIn(i => keyname, :value),
                        Input(:sel => i) => CompIn(i => keyname, :sel),
                        CompOut(i => keyname, :out) => Output(keyname)
                    ) for i=1:m.n_possibilities for keyname in keys_deep(full_out)
                )
            )
    end

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