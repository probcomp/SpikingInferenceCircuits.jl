"""
    is_cpts(dist_or_genfn_or_ir)

Whether the given distribution / generative function / static IR
only uses CPTs and LabeledCPTs.  (Ie. whether this 
has already been compiled at least to LabeledCPTs.)
"""
is_cpts(::CPT) = true
is_cpts(::LabeledCPT) = true
is_cpts(::Gen.Distribution) = false

is_cpts(ir::Gen.StaticIR) = (
    all(is_cpts(node.dist) for node in ir.choice_nodes) &&
    all(is_cpts(node.generative_function) for node in ir.call_nodes)
)

is_cpts(gf::Gen.StaticIRGenerativeFunction) = is_cpts(Gen.get_ir(typeof(gf)))