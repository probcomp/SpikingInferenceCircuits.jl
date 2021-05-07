# `DiscreteIRTransforms`

Transformations on the IR of generative functions where all random samples
are from discrete distributions on finite domains.
In particular, this provides transformations to an equivalent IR,
but where every distribution is a `LabeledCPT`, or to an equivalent IR
but where every distribution is a `CPT` and each value has been mapped
from its original domain to the domain `{1, ..., n}` for some `n`.

Note that currently these transformations only work for a limited class of
models, which capture common patterns in Gen.