# SpikingInferenceCircuits

Library implementing probabilistic inference circuits for Spiking Neural Networks.
The high-level implementations are hardware-agnostic, and could be compiled
to (e.g.) FPGAs or ASICs in the future by defining compilation routes for those targets.
However, currently full compilation paths to primitive components are only provided for
the `Spiking` target.

This library uses the [Circuits library](https://github.com/probcomp/Circuits.jl) for circuit representation & compilation,
and the [Spiking circuits library](https://github.com/probcomp/SpikingCircuits.jl) for primitive spiking components and the SNN simulator.

## Current goals
Our current goals include compiling a subset of [Gen](gen.dev) probabilistic programs into spiking neural
networks in forward-sampling mode, implementing importance sampling using Gen target & proposal distributions,
and eventually implementing sequential monte carlo for these models.

### Setup
After cloning this repo, from the `SpikingInferenceCircuits.jl/` directory:
```zsh
] activate .
] add https://github.com/probcomp/Gen.jl#20210615-marcoct-sml
] add https://github.com/probcomp/Circuits.jl
] add https://github.com/probcomp/SpikingCircuits.jl
] build
```

(I haven't tested this recently; there may be some other work to do to get the envirnoment set up properly.)

### Repository Structure

There are 3 main parts:
- `src` contains the core code
- `runs` contains scripts used for testing components
- `visualization` contains code to produce visualizations of circuits.  Eventually, this should
be replaced with a better visualizer, and perhaps factored into its own repository (or made part of the
Circuits repository).

I'm currently reorganizing and rewriting large parts of `src`, so the imports in `runs` will mostly be wrong
(and some of the runs may become irrelevent / used foor outdated components.)

### `src`

(This is currently WIP.)

An outline of the directory structure is:
```
src/
    CPTs/
    DiscreteIRTransforms/
    ProbEstimates/
    DynamicModels/
    circuits/
        value_types.jl
        pulse_ir/
            primitives/
            poisson_implementations/
        stochastic_digital_circuits/
            primitives/
            pulse_ir_implementations/
        generative_functions/
            composite/
            leaf/
        inference/
```

- `CPTs` a Julia package exposing the `CPT` (conditional probability table)
  and `LabeledCPT` Gen distributions.
- `DiscreteIRTransforms` is a Julia package for transforming IRs for Gen models where
  all variables are discrete and have finite domains.  In particular, it contains some
  transformations to convert from Static IR (+ combinators) generative functions
  to equivalent generative functions where all distributions are CPTs.
- `ProbEstimates/` is a module for running inference in Gen, injecting noise
  into each probability or inverse-probability value used in the calculations.
  This noise mirrors the type of noise which arises in the spiking neural network
  implementations.  This lets us test the robustness of the SNN implementation of
  inference algorithms in Gen.
- `DynamicModels/` is a WIP module providing utility functions for constructing
  and running SMC inference in dynamic models.

- `circuits` contains the code to compile models and inference programs into circuits
  and ultimately into spiking neural networks.  The sub-directory structure is roughly:
  - `circuits/value_types.jl` defines `Value` types (from the Circuits library) used for the circuits
  - `circuits/generative_functions` contains generative function PROPOSE and ASSESS circuits.
  - `circuits/inference` contains code for producing inference circuits.  I haven't thought through
    what this should look like.
  - `circuits/stochastic_digital_circuits/` and `circuits/pulse_ir/` define the stochastic digital circuits
    and Pulse IR primitive components (and maybe some non-primitive components too.  Eventually
    they will also include SDC --> Pulse IR implementations, and Pulse IR --> Poisson neuron implementations.
    `pulse_ir/` will also contain code for satisfying the temporal interfaces of the Pulse IR.

### Visualizations

The visualizer is web-based.  The front-end code is in `visualization/frontend`; this is a `npm` package.
To install the dependencies, run `npm install` from within the `visualization/frontend` folder.
(If you don't have `npm` installed, you can get it [here](https://www.npmjs.com/get-npm).)
To use the visualizer, run an http server from within this folder.  One way to do this is to
get the `npm` package `http-server` by running `npm install -g http-server`, and then running
the `http-server` command from within `visualization/frontend`.

To view a component, it must first be compiled into a format that the frontend understands.
For this, there is the julia script `visualization/component_interface.jl`.
Likewise, there is a julia script `visualization/animation_interface.jl` to compile the output
of the spiking simulator into a JSON file the frontend can use to produce an animation of spiking circuit operation.

See `runs/cpt_sample_score.jl` for examples of how the visualization scripts are used.  The relevant code snippet is:

```julia
includet("../visualization/component_interface.jl")

open("visualization/frontend/renders/cpt.json", "w") do f
    JSON.print(f, viz_graph(circuit), 2)
end
println("Wrote component viz file.")

events = Sim.simulate_for_time_and_get_events(circuit,  16.0;
    initial_inputs=(:in_vals => 1 => 2, :in_vals => 2 => 1)
)
println("Simulation run.")

includet("../visualization/animation_interface.jl")

open("visualization/frontend/renders/cpt_anim.json", "w") do f
    JSON.print(f, animation_to_frontend_format(Sim.initial_state(circuit), events), 2)
end
println("Wrote animation file.")
```
(For this to work, you will have to `mkdir renders` from within `visualization/frontend` first.)

Then, from the website (http://localhost:8080), enter the name of the file (here, `cpt.json`)
to load the visualization.