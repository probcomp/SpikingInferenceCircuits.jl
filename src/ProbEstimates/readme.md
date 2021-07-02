# ProbEstimates

This is a small library which makes it possible to run experiments in Gen using
estimates of the probabilities of samples from distributions,
rather than the exact value.  This can be used to run experiments
with the same type of noise which would arise when implementing
probabilistic inference in hardware with certain characteristics
(e.g. a spiking neural network).

In particular, the library exposes a few generative functions which produce estimates of
their probability densities, rather than the true values:
- `Cat` (categorical)
- `LCat` (labeled categorical)
- `PseudoMarginalizedDist` (a distribution implemented using auxiliary variables which are pseudo-marginalized over)

### Note: this library uses some Gen interfaces incorrectly!

To result in Gen inference algorithms being run with the same type of noise which occurs
in a certain class of stochastic hardware, this library implements the generative function interface
a bit incorrectly for the generative functions it exposes.

Here are the violations I am aware of:

##### Violation 1: `get_score` is nondeterministic
`get_score(tr)` is nondeterministic for a trace from one of these generative functions;
each time it is run, a new probability estimate is drawn.

##### Violation 2: different probability-estimation mechanisms for `propose` and `assess`
Furthermore, while Gen has native support for certain types of pseudo-marginal probability estimates,
the probability estimates used cannot be produced under this interface.  That is because
the way that probability estimates are produced in `propose` mode (where we want an unbiased estimate
of the reciprocal probability) is different from the way that probability estimates
are produced in `assess` mode.  [By ``different'', I mean that the estimators cannot both arise
due to the use of the same auxiliary randomness ``r``, as is required by the strict definition
of the GFI.]

Thus, under the strict interface, the same generative function could not implement both `propose` and
`assess` in the way one of the spiking neural networks would.

One option would be to expose `proposeCat` and `assessCat`, two separate versions of `Cat` which take
probability estimates as is needed for `propose` and `assess`.  Instead, to simplify notation,
I have just implemented one version, `Cat`, which does not strictly satisfy the GFI, but does take estimates
during `propose` and `assess` in the same way the SNNs do.

##### Violation 3: behavior depends on global state
To enable users to control the parameters of probability estimation,
there are global variables maniuplatable by global functions
(like `use_perfect_weights!()` or `use_noisy_weights!()`) which affect how probability
estimates are used by generative functions in this library.
(This violates the property that generative functions should not depend upon any external state.)