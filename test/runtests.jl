# The `test` directory contains some automatic tests;
# `runs` contains scripts for running code, without automatic checks that things look correct
using Test
using Gen
using Circuits
using SpikingCircuits
using SpikingInferenceCircuits
const SIC = SpikingInferenceCircuits

includet("compile_simple_model_test.jl")