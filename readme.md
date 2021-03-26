# SpikingInferenceCircuits

Library implementing probabilistic inference circuits for Spiking Neural Networks.
The high-level implementations are hardware-agnostic, and could be compiled
to (e.g.) FPGAs or ASICs in the future by defining compilation routes for those targets.
However, currently full compilation paths to primitive components are only provided for
the `Spiking` target.

This library uses the [Circuits library]() for circuit representation & compilation,
and the [Spiking circuits library]() for primitive spiking components and the SNN simulator.

## Current goals
Our current goals include compiling a subset of [Gen](gen.dev) probabilistic programs into spiking neural
networks in forward-sampling mode, implementing importance sampling using Gen target & proposal distributions,
and eventually implementing sequential monte carlo for these models.

### Repository Structure

There are 3 main parts:
- `src` contains the core code
- `runs` contains scripts used for testing components
- `visualization` contains code to produce visualizations of circuits.  Eventually, this should
be replaced with a better visualizer, and perhaps factored into its own repository (or made part of the
Circuits repository).

### `src`

This is the main code for compiling inference circuits from Gen to hardware.
- `value_types.jl` defines `Value` types (from the Circuits library) used for the circuits
- `cpt.jl` defines a conditional probability table
- `components` defines various components used in inference circuits.  Most of these are defined
  as abstract components, with a `Spiking`-specific implementation which enables the abstract circuits
  to be compiled for `Spiking` hardware.  There are also a few Spiking-specific components.
- `compiler` contains code to compile from Gen to circuits which implement
  some inference/sampling functionality (e.g. some [generative function interface](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1) methods, like `propose`).  The circuits it compiles to are implemented
  using the components from the `components` directory.

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