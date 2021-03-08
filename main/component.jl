using LightGraphs

######################
# Abstract interface #
######################

"""
    abstract type Component end

A circuit component.
"""
abstract type Component end

"""
   abstract type GenericComponent <: Component end

A component representation which is not primitive for a target, nor composed
of other components (though it may be able to be _implemented_ using other components).
"""
abstract type GenericComponent <: Component end

"""
    abstract type PrimitiveComponent{T} <: Component end

A component primitive to target `T <: Target`.  Ie. the simulator for that target
can operate directly on this component.
"""
abstract type PrimitiveComponent{Target} <: Component end

"""
    inputs(::Component)::CompositeValue

A `CompositeValue` giving the inputs to this component.
"""
inputs(::Component)::CompositeValue = error("Not implemented.")

"""
    outputs(::Component)::CompositeValue

A `CompositeValue` giving the outputs from this component.
"""
outputs(::Component)::CompositeValue = error("Not implemented.")

"""
    abstract(c::Component)

The more abstract component which was implemented to yield `c`, or `nothing` if such a component does
not exist or is not available.
"""
abstract(::Component) = nothing

### TODO: document `implement` (and `is_implementation_for` and `implement_deep`)
### TODO: document `target` (and maybe add an `is_concrete`?)
# TODO: while working on this, think about whether things are set up right or this could be done better

######################
# CompositeComponent #
######################

"""
    abstract type NodeName end

The name of a node in a `CompositeComponent` graph.
Is either an `Input` to the component, an `Output` from it,
or an input to/output from a sub-component of the composite component
(a `CompIn` or `CompOut`).

In a `CompIn` or `CompOut`, sub-components in this `c::CompositeComponent` are named using their name in `c`.

In any `NodeName`, an input/output value for a component `c`
(where `c` is either a top-level component or a subcomponent)
may either directly be a value in `inputs(c)` or `outputs(c)`, or may be a value nested
within a `CompositeValue` in `inputs(c)` or `outputs(c)`.
A value in `inputs(c)` or `outputs(c)` is referred to using the name within the `inputs`/`outputs`
`CompositeValue`.
A nested value is referred to using a linked-list
`n_1 => (n_2 => ... (n_{n-1} => n_n))`.

(Eg. `Inputs(x => (y => z))` refers to `inputs(c)[x => y => z] == inputs(c)[x][y][z]`.)
"""
abstract type NodeName end

"""
    Input <: NodeName
    Input(name)

An input value to a composite component with the given `name`.
"""
struct Input{ID} <: NodeName
    id::ID
end
"""
    Output <: NodeName
    Output(name)

An output value from a composite component with the given `name`.
"""
struct Output{ID} <: NodeName
    id::ID
end
"""
    CompIn <: NodeName
    CompIn(comp_name, in_name)

An input value to a sub-component named `comp_name` in a composite component,
with name `in_name` in the subcomponent.
"""
struct CompIn{I1, I2} <: NodeName
    comp_name::I1
    in_name::I2
end
"""
    CompOut <: NodeName
    CompOut(comp_name, out_name)

An output value from a sub-component named `comp_name` in a composite component,
with name `out_name` in the subcomponent.
"""

struct CompOut{I1, I2} <: NodeName
    comp_name::I1
    out_name::I2
end

# TODO: better docstring here?
"""
    CompositeComponent <: Component

A `Component` represented as a graph of wires connecting inputs & outputs of subcomponents.
"""
# Has an input `CompositeValue`, an output `CompositeValue`, and some number of internal `Component`s.
# Each input/output `Value` and each `Component` has a name.
# There is a graph on the input/output sub-Values and all the input/output Values for the internal components.
# A `NodeName` is thus one of:
# 1. An `Input` name
# 2. An `Output` name
# 3. A pair `(subcomponent name, subcomponent input name)`
# 4. A pair `(subcomponent name, subcomponent output name)`

# We store a 2-way mapping between these node names and vertex indices in the graph.
# We store the `Value` for each input/output name, and the `Component` for each component name.
# """
struct CompositeComponent{InID, OutID, SC} <: Component
    input::CompositeValue{InID}
    output::CompositeValue{OutID}
    subcomponents::SC # `Tuple` or `NamedTuple` of subcomponents
    node_to_idx::Dict{NodeName, UInt} # maps from `NodeName` to index of this node in the graph
    idx_to_node::Vector{NodeName} # maps index of a node in the graph to its `NodeName`
    graph::SimpleDiGraph # graph of connections between input/output values
    abstract::Union{Nothing, Component} # the component which was `implement`ed to obtain this `CompositeComponent`
    
    # ensure `subcomponents` is either a Tuple or NamedTuple of Components
    CompositeComponent(
        input::CompositeValue{InID},
        output::CompositeValue{OutID},
        subcomponents::SC,
        node_to_idx::Dict{NodeName, UInt},
        idx_to_node::Vector{NodeName},
        graph::SimpleDiGraph,
        abstract::Union{Nothing, Component}=nothing
    ) where {InID, OutID, SC <: Union{
        Tuple{Vararg{<:Component}}, 
        NamedTuple{<:Any, <:Tuple{Vararg{<:Component}}}
    }} = new{InID, OutID, SC}(input, output, subcomponents, node_to_idx, idx_to_node, graph, abstract)
end

"""
    CompositeComponent(
        input::CompositeValue, output::CompositeValue,
        subcomponents::Union{Tuple, NamedTuple}, edges, abstract=nothing
    )

A composite component with the given inputs, outputs, subcomponents, graph edges, and abstract version.
If `subcomponents` is a `Tuple`, the subcomponent names will be `1, ..., length(subcomponents)`.
If `subcomponents` is a `NamedTuple`, the subcomponent names will be the keys in the named tuple.
`edges` should be an iterator over `Pair{<:NodeName, <:NodeName}`s of the form `src_nodename => dst_nodename`,
giving the inputs/outputs of the component and its subcomponents to connect with an edge.  (Each `src_nodename`
should be either an `Input` or `CompOut`, and each `dst_nodename` should be a `Output` or `CompIn`.)
"""
function CompositeComponent(
    input::CompositeValue,
    output::CompositeValue,
    subcomponents,
    edges, abstract::Union{Nothing, Component}=nothing
)
    idx_to_node = collect(Iterators.flatten((
        (Input(k) for k in keys(input)), (Output(k) for k in keys(output)),
        (CompIn(compname, inname) for (compname, subcomp) in pairs(subcomponents) for inname in keys(inputs(subcomp))),
        (CompOut(compname, outname) for (compname, subcomp) in pairs(subcomponents) for outname in keys(outputs(subcomp)))
    )))
    node_to_idx = Dict{NodeName, UInt}(name => idx for (idx, name) in enumerate(idx_to_node))
    
    graph = try
        SimpleDiGraphFromIterator((Edge(node_to_idx[src_name], node_to_idx[dst_name]) for (src_name, dst_name) in edges))
    catch e
        if e isa KeyError && e.key isa NodeName
            @error("$(e.key) used in `edges` but does not match the nodenames derived from `input`, `output`, and `subcomponents`")
        end
        throw(e)
    end
    @assert nv(graph) == length(idx_to_node)

    CompositeComponent(input, output, subcomponents, node_to_idx, idx_to_node, graph, abstract)
end
inputs(c::CompositeComponent) = c.input
outputs(c::CompositeComponent) = c.output
abstract(c::CompositeComponent) = c.abstract

"""
    does_output(c::CompositeComponent, name::Union{Input, CompOut})

True if the node with the given name is connected to an `Output`; false false otherwise.
"""
# TODO: should we try to be smart about whether we iterate through the neighbors or the outputs?
does_output(c::CompositeComponent, name::Union{Input, CompOut}) =
    any(
        c.idx_to_node[idx] isa Output
        for idx in neighbors(c.graph, c.node_to_idx[name])
    )

"""
    receivers(c::CompositeComponent: name::Union{Input, CompOut})

Iterator over all the `NodeName`s which receive output from the node named `name`.

(Each element of the outputted iterator will either be an `Output` or a `CompIn`.)
"""
receivers(c::CompositeComponent, name::Union{Input, CompOut}) = (
        c.idx_to_node[idx] for idx in neighbors(c.graph, c.node_to_idx[name])
    )


# TODO: is_implementation_for & implement_deep
# implement_deep(c::PrimitiveComponent{U}, t::Target) where {U <: Target} = error("Cannot implement $c, a PrimitiveComponent{$u}, for target $t.")
# implement_deep(c::PrimitiveComponent{<:T}, ::T) where {T <: Target} = c
# implement_deep(c::GenericComponent, t::Target) = implement_deep(implement(c, t), t)
# implement_deep(c::CompositeComponent, t::Target) =
#     if is_implementation_for(c, t)
#         c
#     else
#         CompositeComponent(
#             # TODO!
#         )
#     end


# TODO: figure out the details of Junction.  Do we want one `Junction` concrete type, or instead
# something like a `Junction` abstract type with concrete subtypes `ValSplit`, `ValMerge`, and some combiner?
"""
    Junction <: PrimitiveComponent{Target}

A Junction reroutes Values, for instance joining values together or splitting a Value into several.
It is a primitive component for every target.

Implementation & details of how this works: TODO
"""
struct Junction <: PrimitiveComponent{Target}

end