# Spiking Neural: Main
Core spiking neural network research code!

### Structure

- `Circuits` is a Julia package for operating on circuits in an abstract way
- `SpikingCircuits` is a Julia package specializing `Circuits` to the Spiking Neural domain.
  It includes primitive units for circuits (eg. values like `SpikeWire` and components like `PoissonNeuron`),
  and implements a SNN simulator `SpikingSimulator`.
- `visualizations` contains code to produce visualizations, including:
  - Spiketrain visualization
  - Web-based visualizations of circuits and animations of their behavior
    - `visualizations/circuit_visualization/animation_interface.jl` and `visualizations/circuit_visualization/component_interface.jl` are scripts to compile circuits into the format the web interface understands
    - `visualizations/circuit_visualization/frontend` contains the web front-end code.  To run the webpage, launch a web server from this folder (eg. using the npm package `http-server`).
- `components` defines some value types and components per the `Circuits` interface
- `runs` contains scripts to simulate these components, and produce visualizations and other debugging results

This is all very much under development.