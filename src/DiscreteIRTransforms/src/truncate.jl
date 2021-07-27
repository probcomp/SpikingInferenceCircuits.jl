# TODO: generic map methods over static IR leaf nodes

function truncate(ir::StaticIR, minprob)
    builder = StaticIRBuilder()
    for node in ir.nodes
        new_node = truncate(node, minprob)
        add_node!(builder, new_node)
        if ir.return_node.name == new_node.name
            Gen.set_return_node!(builder, new_node)
        end
    end
    return build_ir(builder)
end
truncate(gf::StaticIRGenerativeFunction, minprob) = to_gf(
    truncate(get_ir(gf), minprob),
    add_gf_name_suffix(gf, "truncated_dists")
)

truncate(node::ArgumentNode, _) = node
truncate(node::JuliaNode, _)    = node
truncate(node::RandomChoiceNode, minprob) =
    @set node.dist = truncate(node.dist, minprob)
truncate(node::GenerativeFunctionCallNode, minprob) =
    @set node.generative_function = truncate(node.generative_function, minprob)

truncate(lcpt::LabeledCPT, minprob) = @set lcpt.cpt = truncate(lcpt.cpt, minprob)
truncate(cpt::CPT, minprob) =
    CPT(
        [
            truncate_dist(dist, minprob)
            for dist in cpt.dists
        ]
    )
truncate_dist(dist::CPTs.Categorical, minprob) =
    CPTs.Categorical(truncate_dist(CPTs.probs(dist), minprob))
function truncate_dist(pvec::Vector, minprob)
    # println("in truncate_dist; minprob = $minprob")
    if !isapprox(sum(pvec), 1.)
        error("pvec = $pvec is not a probability vector.  minprob = $minprob")
    end

    mininvec = minimum(p for p in pvec if p != 0)
    if mininvec ≥ minprob
        return pvec
    else
        first_to_truncate = findfirst(pvec .== mininvec)
        return truncate_dist(
            normalize([i == first_to_truncate ? 0. : p for (i, p) in enumerate(pvec)]),
            minprob
        )
    end
end
normalize(vec) = vec/sum(vec)