"""
    Mux <: GenericComponent
    Mux(n_possibilities, out_value)

Multiplexer.  Recieves `n_possibilities` values in the `:values => i` inputs,
and a `1` in one of `n` `:sel => i` Binary inputs.
Outputs the `:values => i` input for the selected `i`.

To implement this for a specific target where the values `compiles_to_binary`,
it suffices to implement `BitMux`, and a default implementation of `Mux` in terms of this can be used.
"""
struct Mux <: GenericComponent
    n_possibilities::Int
    out_value::CompositeValue
end

Circuits.outputs(m::Mux) = m.out_value
Circuits.inputs(m::Mux) = NamedValues(
    :values => IndexedValues(m.out_value for _=1:m.n_possibilities),
    :sel => IndexedValues((Binary() for _=1:m.n_possibilities), FiniteDomainValue(m.n_possibilities))
)

### Implement MUX for a FiniteDomainValue encoded into binary using a one-hot encoding
### by using a bunch of BitMuxes which pass through each bit separately.
### TODO: Generalize this, e.g., to the case where the value is encoded in
### Base 2 rather than one-hot.

# This is not well documented and is not the general interface a `Mux` satisfies
# in digital logic--so I should improve the naming, etc.

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


# For any target, we can implement a Mux using the `BitMux` for the 
Circuits.implement(m::Mux, t::Target) =
    if can_implement(BitMux(), t) && compiles_to_binary(outputs(m), t)
        _bit_mux_implementation(m, t)
    else
        error("Mux not implemented for $t.  If $t implemented BitMux(), we could implement in terms of this.")
    end

_bit_mux_implementation(m::Mux, t::Target) =
    let full_in = implement_deep(inputs(m), t),
        full_out = implement_deep(outputs(m), t)
            CompositeComponent(
                full_in, full_out,
                map(bit_muxes, full_in[:values].vals),
                Iterators.flatten(
                    (
                        Input(:values => i => keyname) => CompIn(i => keyname, :value),
                        Input(:sel => i) => CompIn(i => keyname, :sel),
                        CompOut(i => keyname, :out) => Output(keyname)
                    ) for i=1:m.n_possibilities for keyname in keys_deep(full_out)
                )
            )
    end

bit_muxes(v::CompositeValue) = ComponentGroup(map(bit_muxes, v.vals))
bit_muxes(::Binary) = BitMux()
bit_muxes(v::Value) = bit_muxes(abstract(v)) # eg. SpikeWire -> Binary