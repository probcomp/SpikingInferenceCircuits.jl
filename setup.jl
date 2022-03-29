using Pkg
# Pkg.activate(".")
try
    Pkg.rm("Circuits")
catch; end;
try
    Pkg.rm("SpikingCircuits")
catch; end;
Pkg.add(url="https://github.com/femtomc/CircuitViz.jl")
Pkg.add(url="git@github.com:probcomp/Circuits.jl.git")
Pkg.add(url="git@github.com:probcomp/SpikingCircuits.jl.git")
Pkg.develop(path="src/CPTs")
Pkg.develop(path="src/DiscreteIRTransforms")
Pkg.build()