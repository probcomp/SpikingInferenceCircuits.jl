import JSON

x_align_constraint(node_indices) =
    Dict(
        "type" => "alignment",
        "axis" => "x",
        "offsets" => [
            Dict(
                "node" => idx,
                "offset" => 0
            )
            for idx in node_indices
        ]
    )
function maybe_push_x_constraints!(constraints, indices)
    consts = x_align_constraint(indices)
    if length(consts) > 1
        push!(constraints, consts)
    end
end

function json_graph(comp::CompositeComponent)
    nodes = [
        Dict("name" => "$(nodename)")
        for nodename in comp.idx_to_node
    ]
    links = [
        Dict("source" => e.src, "target" => e.dst)
        for e in edges(comp.graph)
    ]
    groups = []
    constraints = []

    for (subcomp_name, subsc) in pairs(comp.subcomponents)
        comp_val_indices(Name, getval) = [
            comp.node_to_idx[Name(subcomp_name, name)]
            for name in keys(getval(subsc))
        ]

        comp_in_indices = comp_val_indices(CompIn, inputs)
        comp_out_indices = comp_val_indices(CompOut, outputs)

        push!(groups, Dict("leaves" => vcat(comp_in_indices, comp_out_indices)))
        maybe_push_x_constraints!(constraints, comp_in_indices)
        maybe_push_x_constraints!(constraints, comp_out_indices)
    end

    JSON.json(Dict(
        "nodes" => nodes,
        "links" => links,
        "groups" => groups,
        "constraints" => constraints
    ))
end