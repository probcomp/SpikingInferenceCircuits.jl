# Currently this file is not included by default, and must be included separately.
# It allows models defined using `Cat` and `LCat` from `ProbEstimates`
# to be compiled using `DiscreteIRTransforms`.

# TODO: encapsulate these definitions in a somewhat less ad-hoc way!

DiscreteIRTransforms.get_domain(l::ProbEstimates.LCat) = l.labels
DiscreteIRTransforms.assmt_to_probs(::ProbEstimates.LCat) = ((pvec,),) -> pvec