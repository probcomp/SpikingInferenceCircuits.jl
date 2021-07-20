import Pkg;
Pkg.activate(".");
Pkg.build();
using Revise;

# Fill in with whatever run file you want!:
include("experiments/velwalk1d/snn_run.jl")