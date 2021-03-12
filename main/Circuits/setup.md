TODO: should I change from "abstract" to something else?  As is, abstract and concrete are not opposites (an
abstract component may also be concrete.)

## Components and Values

A `Component` is an element which can process information.
Information processing occurs along 2 dimensions, which we will call ``space''
and ``time''.  Space may be multi-dimensional; time is 1 dimensional (and may be either discrete or continuous).
In some cases, ``space'' is the physical space a circuit is layed out in, and ``time'' is physical time.
However, this isn't always the case.  We may target software simulation, in which case ``space'' also includes
the physical time it takes to multiplex different parts of a simulated circuit into our computer's CPU, and ``time''
really refers to the simulated time (not physical time).

A component accepts input `Value`s and outputs `Value`s.  A `Value` should be thought of
not as a concrete value of some sort like `5` or `true`, but instead as the potential for some occurance to happen in
time which we indicates one of these concrete values.  As an example, we may say that a digital adder circuit
takes 2 input `Value`s and has 1 output `Value`, since at any given timestep of the computer,
2 concrete values may be input, and one value may be output.  The output `Value` is analagous to the set of wires
conveying the output from the circuit.  It is not analagous to the concrete numbers (like `5`) which are indicated
when certain voltage patters occur in those output wires--nor is it analogous to any given voltage pattern.

Broadly, any component or value may be classified as either *generic*, *primitive*, or *composite*.

### Generic
Generic Values are a mathematical description of a certain type of value which can be conveyed via some occurance in time.
Generic Components are likewise purely mathematical description of an operation
which can be performed on values.
In practice, users need not be able to rigorously describe the mathematical model for
generic Components or Values.  They simply use generic Values and Components to build information-flow
graphs to describe, in an abstract way, a computational process.

#### Examples of generic Values include:
1. `FiniteDomainValue(n)`: conveys a value from the set `{1, ..., n}`
2. `Real`: conveys a real number

#### Examples of generic components include:
1. `CPTSampler(CPT)`: Samples from a CPT.  Inputs: a `FiniteDomainValue` for each CPT input.  Output: one `FiniteDomainValue`.
2. `CalculateISWeight(model, proposal)`: Outputs the IS weight under a certain model and proposal.
Possible Inputs: `FiniteDomainValue`s for each variable in the model and each proposed variable.  Outputs: `Real`.

### Primitive
A Primitive Value describes the potential for a concrete value to be conveyed
which can be implemented by a simulator or physical process.  For instance,
a `VoltageWire` is a primitive value to a circuit, which at any given time
is conveying the concrete value of the electric potenaial relative to ground.
In a spiking neural network, a `SpikeWire` is a primitive value, which at any time
either conveys the value `isSpiking` or `isNotSpiking`.

A primitive component is likewise a
unit which can be simulated or implemented physically; it operates on primitive values.
Each primitive value is primitive for some *target*, specifying the type of hardware or simulation
for which this is a given.

#### Examples
The primitive value for the `Spiking` target is a `SpikeWire` which at any time either is spiking or is not spiking.
We can implement different components for the `Spiking` target, such as:
1. LIF Neuron
2. Poisson process neuron
3. ...
All of these accept `SpikeWire` inputs and output `SpikeWire`s.

### Composite
A composite value is a value made from some number of other values.  (We can think of this of a tuple of other values.)
A composite component is a component described as a multigraph of other components.
We can view this multigraph as a graph where:
- The nodes comprise:
  - Each input to the composite component
  - Each output from the composite component
  - Each input to a sub-component
  - Each output from a sub-component
- If edge `(a, b)` is part of the graph, then:
  - `a` is either:
    - an input to the composite component
    - an output from a sub-component
  - `b` is either:
    - an input to a sub-component
    - an output from the composite component
  - The `Value` output from `a` is the same as the `Value` input to `b`

### Targets
A `Target` is an environment which can perform computation using components.
For instance, a `Target` may be a type of simulator or hardware.

### Implementable & Concrete Values and Components
We say that a `Value` is *implementable* for a given target if there is at least one way
to implement it as a composite Value of Primitive Values for that target.

We say a `Component` is *implementable* for a given target if there is at least one
way to implement it as a composite Component of Primitive Components for that target.

We say that a `Value` or `Component` is *concrete* if it is only implementable for exactly one
target, and there is *exactly one*
way to implement it as a composite Value/Component of primitives for that target.
(Eg. the spiking Categorical Value representation is concrete.)
Every primitive Value and Component is concrete.
We sometimes say that a value or component is *abstract* if it is not concrete.

## Designing Components
To simulate a circuit, we must know:
1. A `Target` -- in this case we will assume this is a simulator for something
2. The primitive `Value`s for the target
3. A way to implement the operation of each primitive component given the time-signal for the input Values
4. A way to implement the operation of a composite component comprising only primitive components.  This implementation
  must be such that any communication between 2 sub-components occurs via a `Value`.

To design a primitive `Component`, we must implement it according to the simulator/hardware platform
we want to use.

#### Designing a nonprimitive `Component`
When designing a nonprimitive `Component`, we must first decide what types of `Value`s it operates on
(and possibly create new `Value` types if needed).
We must then understand what our sub-components we will build the component in terms of.
If we want to ultimately be able to simulate the component, we must make sure the `Value` and
sub-`Component` types we use are implementable for our given target.

To design the component, we define the new value types we need, describe
the component's interface, and possibly provide an implementation.  

To provide an implementation, we simply provide a graph of subcomponents which yields
the correct type of information flow.

## Simulating Components
As described above, we assume that a simulator knows how to implement primitive components
and graphs of primitive components.

Thus the question of how to simulate a component in general boils down to how to implement
it as a graph of primitives.  If a component is concrete, this is simple, since there is one
way to implement like this.  If a component is implementable but not concrete, the user must provide
a specification for which implementation of this to use.  (This both includes
what concrete Values to compile abstract Values to, and what concrete graphs of concrete components to use.)