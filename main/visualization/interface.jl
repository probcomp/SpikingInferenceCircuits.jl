import JSON

shortname(n::Union{Input, Output}) = "$(n.id)"
shortname(n::CompIn) = "$(n.in_name)"
shortname(n::CompOut) = "$(n.out_name)"

x_ineq_constraints(leftidx, rightidx; equality=false) =
    Dict(
        "type" => "separation",
        "axis" => "x",
        "left" => leftidx,
        "right" => rightidx,
        "equality" => equality
        # "gap" filled in by front-end code
    )

function maybe_push_x_align_constraints!(constraints, indices)
    consts = x_align_constraint(indices)
    if length(consts) > 1
        push!(constraints, consts)
    end
end





#######

function json_graph(comp::CompositeComponent)
    nodes = nodes_json(comp)
    links = links_json(comp)
    groups, constraints = [], []
    push!(constraints, io_constraints(comp))
    for (subcomp_name, subcomp) in pairs(comp.subcomponents)
        handle_subcomponent!(groups, constraints, subcomp_name, subcomp)
    end

    JSON.json(Dict(
        "nodes" => nodes,
        "links" => links,
        "groups" => groups,
        "constraints" => constraints
    ))
end


nodes_json(comp) = [
        Dict("name" => shortname(nodename))
        for nodename in comp.idx_to_node
    ]
links_json(comp) = [
        # subtract 1 to move to javascript 0-indexing
        Dict("source" => e.src - 1, "target" => e.dst - 1)
        for e in edges(comp.graph)
    ]
io_constraints(comp) = 
    let in_indices = js_indices(comp, Input, keys(inputs(comp))),
        out_indices = js_indices(comp, Input, keys(outputs(comp)))
            Iterators.flatten((
                (x_align_constraint(in_indices), x_align_constraint(out_indices)),
                y_order_constraints(in_indices),
                y_order_constraints(out_indices)
            ))
    end

js_indices(comp, Name, names) = [comp.node_to_idx[Name(name)] for name in names] .- 1

x_align_constraint(js_indices) = Dict(
        "type" => "alignment",
        "axis" => "x",
        "offsets" => [
            Dict(
                "node" => idx,
                "offset" => 0
            )
            for idx in js_indices
        ]
    )

y_order_constraints(js_indices) = (
        Dict(
            "type" => "separation",
            "axis" => "y",
            "left" => js_indices[i],
            "right" => js_indices[i + 1],
            "gap" => 0
        )
        for i=1:length(js-indices)-1
    )

function handle_subcomponent!(groups, constraints, subcomp_name, subcomp)

end



######

    for (subcomp_name, subsc) in pairs(comp.subcomponents)
        comp_val_indices(Name, getval) = [
            comp.node_to_idx[Name(subcomp_name, name)]
            for name in keys(getval(subsc))
        ]

        # subtract 1 to move to javascript 0-indexing
        comp_in_indices = comp_val_indices(CompIn, inputs) .- 1
        comp_out_indices = comp_val_indices(CompOut, outputs) .- 1

        push!(groups, Dict("leaves" => vcat(comp_in_indices, comp_out_indices)))
        maybe_push_x_align_constraints!(constraints, comp_in_indices)
        maybe_push_x_align_constraints!(constraints, comp_out_indices)
        
        @assert length(comp_in_indices) > 0 && length(comp_out_indices) > 0 # TODO: add dummy nodes if false
        push!(constraints, x_ineq_constraints(first(comp_in_indices), first(comp_out_indices), equality=true))
        push!(constraints, x_ineq_constraints(first(in_indices), first(comp_in_indices), equality=false))
        push!(constraints, x_ineq_constraints(first(comp_out_indices), first(out_indices), equality=false))
    end
end