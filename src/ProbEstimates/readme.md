# ProbEstimates

This is a small library which makes it possible to run experiments in Gen using
estimates of the probabilities of samples from distributions,
rather than the exact value.  This can be used to run experiments
with the same type of noise which would arise when implementing
probabilistic inference in hardware (e.g. a spiking neural network).