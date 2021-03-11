"""
    visualization/component_interface.jl

Code to convert Julia components into a JSON format renderable
by the WebCola-based front-end.

Doing this conversion server-side rather than client-side may increase
the amount of data we need to send to the client (since the format WebCola wants may not
be very efficient), though I have not tested this.
It would probably be a bit better (in terms of amount of data sent to client,
and due to the desirability of pushing all visualization logic to the client-side)
to have a format in which we directly communicate components,
and have this conversion-to-webcola-format happen in javascript on the client-side.
"""

############
# VizGraph #
############

"""
    VizGraph(nodes, links, groups, constraints)

A data structure containing specifying a visualization of a component.
Heirarchically groups input and output nodes and constrains the nodes to be aligned properly.
This data structure is easily convertible into a JSON format which the front-end accepts (via `JSON.json`).
"""
struct VizGraph{N,L,G,C}
    nodes::N
    links::L
    groups::G
    constraints::C
end

"""
    append_data!(vg1::VizGraph, vg2::VizGraph)

Append the `groups`, `nodes`, `constraints`, and `links`
from `vg2` onto those in `vg1`.
"""
function append_data!(vg1::VizGraph, vg2::VizGraph)
    append!(vg1.groups, vg2.groups)
    append!(vg1.nodes, vg2.nodes)
    append!(vg1.constraints, vg2.constraints)
    append!(vg1.links, vg2.links)
end

to_dict(vg::VizGraph) = Dict(
    "nodes" => vg.nodes,
    "links" => vg.links,
    "groups" => vg.groups,
    "constraints" => vg.constraints
)

#############
# viz_graph #
#############

"""
    vizgraph::VizGraph = viz_graph(comp::Component)

Get a visualizable graph for the component in the format used by the front-end.
"""
viz_graph(comp::Component) = _viz_graph(comp, 0, 0)

#=
    _viz_graph must return such that:

    - the first element of `groups` is a group for the top-level component
    - the first elements of `nodes` are the inputs to the top-level component
    - the first elements of `nodes` after the top-level inputs are the outputs to the top-level component
=#

"""
    vg = _viz_graph(comp::Component, start_node_idx, start_group_idx)

Get a VizGraph for the component, where all references to nodes are offset
by `start_node_idx` and all references to groups are offset by `start_group_idx`.
(So the node in `vg.nodes[1]` should be referred to as `start_node_idx`; `vg.nodes[2]` by `start_node_idx + 1`, etc.)

The vizgraph must obey the following:
- The first element of `vg.groups` is the top-level group for `comp`
- The first elements of `vg.nodes` are the inputs to `comp`
- The first elements of `vg.nodes` after the inputs to `comp` are the outputs to `comp`
"""

# to draw a non-composite component, we just draw its inputs and outputs
_viz_graph(comp::Component, startidx, a; compname=nothing) =
    VizGraph(get_io_nodes(comp), [], [make_comp_group(comp, startidx; compname)], io_constraints(comp, startidx))

# to draw a composite component, we draw the inputs and outputs,
# recursively draw the subcomponents,
# and then draw the links
function _viz_graph(comp::CompositeComponent, start_node_idx, start_group_idx; compname=nothing)
    io_nodes = get_io_nodes(comp)

    # recursively call on subcomponents
    (vizgraph, name_to_nodeidx, name_to_groupidx) = get_subcomps_vizgraph(
        comp,
        start_node_idx + length(io_nodes),
        start_group_idx + 1
    )

    add_io_nodes_to_table!(name_to_nodeidx, comp, start_node_idx)
   
    # i/o nodes go at the beginning
    prepend!(vizgraph.nodes, io_nodes)

    @assert all(
        vizgraph.nodes[idx + 1 - start_node_idx]["name"] == "$(valname(nodename))"
        for (nodename, idx) in name_to_nodeidx
    )

    # construct group which has all the ins/outs as nodes, and all the subcomponents as sub-groups
    insert!(vizgraph.groups, 1, make_comp_group(comp, start_node_idx, values(name_to_groupidx); compname))

    # links
    prepend!(vizgraph.links, make_links(comp, name_to_nodeidx))

    new_constraints = vcat(
        # i/o x align constraints; i/o y order constraints
        io_constraints(comp, start_node_idx),
        # constraints to keep subgroups in the middle of the i/o
        io_on_outside_constraints(comp, name_to_nodeidx, vizgraph.nodes, start_node_idx)
    )
    prepend!(vizgraph.constraints, new_constraints)

    return vizgraph
end

"""
    (vg, name_to_nodeidx, name_to_groupidx) = get_subcomps_vizgraph(comp::CompositeComponent, start_node_idx, start_group_idx)

Get `vg`, a `VizGraph` visualizing all the subcomponents.
Also return `name_to_nodeidx`, a dictionary mapping `NodeName`s for `comp` to the index for that node
in vizgraph for `comp` (which this method does not construct),
and `name_to_groupidx`, a dictionary mapping subcomponent names to the index
of the corresponding subcomponent group in the vizgraph for `comp`.

Note that these dictionaries map to indices in Javascript's 0-indexing scheme, not Julia's 1-indexing scheme.
"""
function get_subcomps_vizgraph(comp::CompositeComponent, start_node_idx, start_group_idx)
    vg = VizGraph([], [], [], [])
    name_to_groupidx = Dict()
    name_to_nodeidx = Dict()
    for (sname, sc) in pairs(comp.subcomponents)
        subcomp_data = _viz_graph(sc, start_node_idx, start_group_idx; compname=sname)
        in_indices = in_inds(sc, start_node_idx)
        out_indices = out_inds(sc, in_indices, start_node_idx)

        for (name, idx) in zip(keys_deep(inputs(sc)), in_indices)
            name_to_nodeidx[CompIn(sname, name)] = idx
        end
        for (name, idx) in zip(keys_deep(outputs(sc)), out_indices)
            name_to_nodeidx[CompOut(sname, name)] = idx
        end
        name_to_groupidx[sname] = start_group_idx

        append_data!(vg, subcomp_data)

        start_node_idx += length(subcomp_data.nodes)
        start_group_idx += length(subcomp_data.groups)
    end

    (vg, name_to_nodeidx, name_to_groupidx)
end

"""
    add_io_nodes_to_table!(name_to_nodeidx, comp, start_node_idx)

Add entries for the `Input` and `Output` nodenames to `name_to_nodeidx`.
"""
function add_io_nodes_to_table!(name_to_nodeidx, comp, start_node_idx)
    idx = start_node_idx
    for name in keys_deep(inputs(comp))
        name_to_nodeidx[Input(name)] = idx
        idx += 1
    end
    for name in keys_deep(outputs(comp))
        name_to_nodeidx[Output(name)] = idx
        idx += 1
    end
    name_to_nodeidx
end 

"""
    in_inds(comp, startidx)

The range of vizgraph indices for the component's inputs.
"""
in_inds(comp, startidx) = startidx:(startidx + length_deep(inputs(comp)) - 1)
"""
    out_inds(comp, startidx)
    out_inds(comp, in_indices, startidx)

The range of vizgraph indices for the component's outputs.
"""
out_inds(comp, in_indices, startidx) = (startidx + length(in_indices)):(startidx + length(in_indices) + length_deep(outputs(comp)) - 1)
out_inds(comp, startidx) = out_inds(comp, in_inds(comp, startidx), startidx)
"""
    io_inds(comp, startidx)

The range of vizgraph indices for the component's inputs and outputs.
"""
io_inds(comp, startidx) =
    let i = in_inds(comp, startidx)
        let o = out_inds(comp, i, startidx)
            first(i):last(o)
        end
    end

########################
# Nodes, groups, links #
########################

"""
    get_io_nodes(comp)

A list of `node` objects for the component's inputs and outputs.
Each object is a dictionary ready to be JSON-ified and sent to the front-end.
"""
get_io_nodes(comp) = collect(Iterators.flatten((
    (Dict("name" => "$nodename", "is_output" => false) for nodename in keys_deep(inputs(comp))),
    (Dict("name" => "$nodename", "is_output" => true) for nodename in keys_deep(outputs(comp)))
)))

comp_type_name(comp) = typeof(comp).name.name
"""
    make_comp_group(comp, initial_node_idx, subgroup_indices=[]; compname)

A group for the component `comp` called `compname` by its parent,
with subcomponents having group indices `subgroup_indices`,
in a format ready to be JSON-ified and sent to the frontend.
"""
function make_comp_group(comp, initial_node_idx, subgroup_indices=[]; compname)
    d = Dict(
        "leaves" => collect(io_inds(comp, initial_node_idx)),
        "comptype" => comp_type_name(comp),
        "name" => compname
    )
    if !isempty(subgroup_indices)
        d["groups"] = collect(subgroup_indices)
    end
    d
end


"""
    make_links(comp, name_to_nodeidx)

The list of graph edges for `comp`,
in a format ready to be JSON-ified and sent to the frontend.
"""
make_links(comp, name_to_nodeidx) = [
        Dict(
            "source" => name_to_nodeidx[comp.idx_to_node[e.src]],
            "target" =>  name_to_nodeidx[comp.idx_to_node[e.dst]],
        )
        for e in edges(comp.graph)
    ]

################
# Constraints #
###############

"""
    io_constraints(comp, startidx)

The constraints to align input and output nodes vertically,
put the inputs and output nodes in the correct vertical order,
and keep the inputs to the left of the outputs.

In a format ready to be JSON-ified and sent to the frontend.
"""
io_constraints(comp, startidx) = 
    let in_indices = in_inds(comp, startidx),
        out_indices = out_inds(comp, in_indices, startidx)
            collect(Iterators.flatten((
                (
                    x_align_constraint(in_indices),
                    x_align_constraint(out_indices),
                    x_offset_constraint( # outputs to right of inputs
                        first(in_indices),
                        first(out_indices)
                    )
                ),
                y_order_constraints(in_indices),
                y_order_constraints(out_indices)
            )))
    end

"""
    io_on_outside_constraints(comp, name_to_nodeidx)

Constraints to ensure subcomponents are in between the inputs and outputs of `comp`.

In a format ready to be JSON-ified and sent to the frontend.
"""
io_on_outside_constraints(comp, name_to_nodeidx, nodes, start_node_idx) = collect(
        Iterators.flatten((
            (
                isempty(pairs(inputs(comp))) ? () : x_offset_constraint( # inputs left of subcomponent inputs
                name_to_nodeidx[Input(first(keys_deep(inputs(comp))))],
                name_to_nodeidx[CompIn(subname, first(keys_deep(inputs(subcomp))))]
                ),
                isempty(pairs(outputs(comp))) ? () : x_offset_constraint( # outputs right of subcomp outputs
                    name_to_nodeidx[CompOut(subname, first(keys_deep(outputs(subcomp))))],
                    name_to_nodeidx[Output(first(keys_deep(outputs(comp))))]
                )
            )
            for (subname, subcomp) in pairs(comp.subcomponents)
        ))
    )


"""
    x_align_constraint(js_indices)

Constraint to vertically align nodes with the given indices.
"""
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

"""
    y_order_constraints(js_indices)

Constraints to keep the given indices in the right vertical order.
"""
y_order_constraints(js_indices) = (
        Dict(
            "type" => "separation",
            "axis" => "y",
            "left" => js_indices[i],
            "right" => js_indices[i + 1],
            "gap" => 0
        )
        for i=1:length(js_indices)-1
    )

"""
    x_offset_constraint(left_js_idx, right_js_idx)

Constraint to keep the node with index `left_js_idx` to the left of that with `right_js_idx`.
"""
x_offset_constraint(left_js_idx, right_js_idx) = Dict(
    "type" => "separation",
    "axis" => "x",
    "left" => left_js_idx,
    "right" => right_js_idx
    # TODO: gap?  probably have this set in javascript?
)