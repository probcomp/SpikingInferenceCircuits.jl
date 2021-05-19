# Setup Instructions

1. Clone this repo.
2. `cd` into the repo. Run `git checkout smc`.
3. Enter the julia repl.  `]activate .`
4. `]rm Circuits`; `rm SpikingCircuits`
5. `]add git@github.com:probcomp/Circuits.jl.git`
6. `]add git@github.com:probcomp/SpikingCircuits.jl.git`
7. `]dev src/CPTs`
8. `]dev src/DiscreteIRTransforms`
9. `]build`

You should now be good to go.