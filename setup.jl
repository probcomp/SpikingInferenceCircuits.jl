using Pkg

macro tryrm(str)
    :( try
         Pkg.rm($str)
       catch; end; )
end

Pkg.activate(".")

@tryrm "Circuits"
@tryrm "SpikingCircuits"
@tryrm "ANNDistributions"
@tryrm "DiscreteIRTransforms"
@tryrm "CPTs"
@tryrm "DynamicModels"
@tryrm "ProbEstimates"
@tryrm "CircuitViz"

Pkg.activate("src/ProbEstimates")

@tryrm "Circuits"
@tryrm "SpikingCircuits"
@tryrm "ANNDistributions"
@tryrm "DiscreteIRTransforms"
@tryrm "CPTs"
@tryrm "DynamicModels"
@tryrm "ProbEstimates"
@tryrm "CircuitViz"

Pkg.develop(path="src/DiscreteIRTransforms")
Pkg.develop(path="src/DynamicModels")

# Pkg.develop(path="src/CPTs")
# Pkg.develop(path="src/ANNDistributions")

# Pkg.develop(path="src/DynamicModels")
# Pkg.develop(path="src/DiscreteIRTransforms/")

Pkg.build()

Pkg.activate(".")

Pkg.add(url="https://github.com/femtomc/CircuitViz.jl")
Pkg.add(url="git@github.com:probcomp/Circuits.jl.git")
Pkg.add(url="git@github.com:probcomp/SpikingCircuits.jl.git")

Pkg.develop(path="src/CPTs")
Pkg.develop(path="src/DiscreteIRTransforms")
Pkg.develop(path="src/DynamicModels")
Pkg.develop(path="src/ProbEstimates")
Pkg.develop(path="src/ANNDistributions")

Pkg.build()