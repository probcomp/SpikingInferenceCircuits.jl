DiscreteIRTransforms.is_cpts(::LCat) = false

# LCat / Cat should be compiled into CPTs, even though this is
# not a distribution
DiscreteIRTransforms.compile_to_primitive(::LCat) = true

DiscreteIRTransforms.get_ret_domain(l::LCat, arg_domains) =
    DiscreteIRTransforms.EnumeratedDomain(labels(l, first(only(arg_domains))))

DiscreteIRTransforms.get_domain(l::LCat, arg_domains) = DiscreteIRTransforms.get_ret_domain(l, arg_domains)
DiscreteIRTransforms.assmt_to_probs(::LCat) = ((pvec,),) -> pvec