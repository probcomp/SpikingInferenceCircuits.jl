struct ValueBlockerPasser{IsBlocker} <: GenericComponent
    value::Value
end
pb(::ValueBlockerPasser{true}) = :block
pb(::ValueBlockerPasser{false}) = :pass
ValuePasser(val) = ValueBlockerPasser{false}(val)
ValueBlocker(val) = ValueBlockerPasser{true}(val)

Circuits.inputs(p::ValueBlockerPasser) = NamedValues(
    pb(p) => Binary(),
    :val => p.value
)
Circuits.outputs(p::ValueBlockerPasser) = NamedValues(
    :out => p.value
)

Circuits.implement(p::ValueBlockerPasser{t}, tar::Target) where {t} =
    let indeep = implement_deep(inputs(p), tar),
        outdeep = implement_deep(outputs(p), tar)
        CompositeComponent(
            indeep, outdeep,
            map(v -> components_for_bits(BitBlockerPasser{t}(), v), outdeep[:out].vals),
            Iterators.flatten(
                (
                    Input(pb(p)) => CompIn(keyname, pb(p)),
                    Input(:val => keyname) => CompIn(keyname, :val),
                    CompOut(keyname, :out) => Output(:out => keyname)
                )
                for keyname in keys_deep(outdeep[:out])
            ), p
        )
    end

components_for_bits(c, ::Binary) = c
components_for_bits(c, v::CompositeValue) = ComponentGroup(map(v -> components_for_bits(c, v), v.vals))
components_for_bits(c, v::Value) = components_for_bits(c, abstract(v)) # eg. SpikeWire -> Binary

struct BitBlockerPasser{IsBlocker} <: GenericComponent end
pb(::BitBlockerPasser{true}) = :block
pb(::BitBlockerPasser{false}) = :pass
Circuits.inputs(b::BitBlockerPasser) = NamedValues(
    pb(b) => Binary(), :val => Binary()
)
Circuits.outputs(b::BitBlockerPasser) = NamedValues(:out => Binary())