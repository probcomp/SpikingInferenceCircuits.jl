### Experiments relating to Pseudo-Marginalization

Plots/experiment TODOs:
1. Plot comparing variance of a low-probability density estimate to latency, for the following setups:
   a. no PM estimation, instead exact marginal density calculation (that might yield low numbers)
   b. naive from-the-prior PM estimation with a couple samples
   c. optimal exact-posterior proposal
   d. some intermediate heuristic proposal

One setup for this would be to extend the "coordinate tracking" demo so that at any time, there's
some probability that there is an erroneous reading.

Implementation TODOs:
1. Gen pseudo-marginalization combinator within the `ProbEstimates` library
2. Support for compiling the pseudo-marginalize combinator in `circuits/generative_functions/`