"""
    Step

A component which advances the discrete time-step in a synchronous circuit's operation.
Upon stepping, outputs the values previously being input into `input`.

(This is analogous to a line of data-flip-flops, in classical circuits.  The abstract
interface does not include the clock wire; that must be added on a hardware-specific basis.)
"""
struct Step <: GenericComponent
    input::Value
end
Circuit.inputs(s::Step) = NamedValues(:in => s.input)
Circuit.outputs(s::Step) = NamedValues(:out => s.input)