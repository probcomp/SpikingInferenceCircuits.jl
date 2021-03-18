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

"""
    target(c::Component)

If `c` is a _concrete component_ which has 1 implementation for 1 target,
returns the `Target` `c` can be implemented for.  Else, returns `nothing`.
"""
target(::Component) = error("Not implemented.  (This may be a non-concrete value with multiple possible targets.)")

"""
    implement(c::Component, t::Target)

Make progress implementing the component for the target, so that a finite
number of repeated calls to `implement` will yield a component `c` so that
`is_implementation_for(c, t)` is true.        
"""
implement(c::Component, t::Target) = no_impl_error(c, t)
implement(c::PrimitiveValue{T1}, t::T2) where {T1 <: Target, T2 <: Target} =
    if T1 <: T2
        c
    else
        error("Cannot implement $c, a PrimitiveComponent{$T1}, for target $t.")
    end

"""
    is_implementation_for(::Component, ::Target)

Whether the given component is an implementation for the given target (ie. whether it can be simulated
by the simulator for that target without any additional implementation work).
"""
is_implementation_for(::PrimitiveComponent{<:T}, ::T) where {T <: Target} = true
is_implementation_for(::PrimitiveComponent, ::Target) = false
is_implementation_for(::GenericComponent, ::Target) = false

"""
    implement_deep(::Component, t::Target)

Implement the component for the target recursively, yielding a component `c` such that
`is_implementation_for(c, t)` is true.
"""
implement_deep(c::PrimitiveComponent{T1}, t::T2) where {T1 <: Target, T2 <: Target} =
    if T1 <: T2
        c
    else
        error("Cannot implement $c, a PrimitiveComponent{$T1}, for target $t.")
    end
implement_deep(c::GenericComponent, t::Target) = implement_deep(implement(c, t), t)

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

Base.:(==)(a::Input, b::Input) = (a.id == b.id)
Base.:(==)(a::Output, b::Output) = (a.id == b.id)
Base.hash(i::Input, h::UInt) = hash(i.id, h)
Base.hash(o::Output, h::UInt) = hash(o.id, h)

# TODO: better docs about what types `comp_name` and `in_name`
# can be, and about the fact that we move nesting to the values
"""
    CompIn <: NodeName
    CompIn(comp_name, in_name)

An input value to a sub-component named `comp_name` in a composite component,
with name `in_name` in the subcomponent.
"""
struct CompIn{I1, I2} <: NodeName
    comp_name::I1
    in_name::I2
    CompIn(i1::I1, i2::I2) where {
        I1 <: Union{Symbol, Integer}, I2 <: Union{Symbol, Integer, Pair}
    } = new{I1, I2}(i1, i2)
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
    CompOut(i1::I1, i2::I2) where {
        I1 <: Union{Symbol, Integer}, I2 <: Union{Symbol, Integer, Pair}
    } = new{I1, I2}(i1, i2)
end

"""
    CompIn(x_1 => ... => x_n, inname)

This is transformed into `CompIn(x_1, x_2 => ... => x_n => inname)`,
so accessing nested components transforms into accessing nested values
in a `ComponentGroup`.
"""
CompIn(p::Pair, inname) = CompIn(p.first, nest(p.second, inname))

"""
    CompOut(x_1 => ... => x_n, outname)

This is transformed into `CompOut(x_1, x_2 => ... => x_n => outname)`.
so accessing nested components transforms into accessing nested values
in a `ComponentGroup`.
"""
CompOut(p::Pair, outname) = CompOut(p.first, nest(p.second, outname))

nest(p, v) = p => v
nest(p::Pair, v) = p.first => nest(p.second, v)

Base.:(==)(a::CompIn, b::CompIn) = (a.comp_name == b.comp_name && a.in_name == b.in_name)
Base.:(==)(a::CompOut, b::CompOut) = (a.comp_name == b.comp_name && a.out_name == b.out_name)
Base.hash(i::CompIn, h::UInt) = hash(i.comp_name, hash(i.in_name, h))
Base.hash(o::CompOut, h::UInt) = hash(o.comp_name, hash(o.out_name, h))

valname(v::Union{Input, Output}) = v.id
valname(v::CompIn) = v.in_name
valname(v::CompOut) = v.out_name

# TODO: better docstring here?
"""
    CompositeComponent <: Component

A `Component` represented as a graph of wires connecting inputs & outputs of subcomponents.

Sub-components may be accessed via Base.getindex, so `c[name]` yields the subcomponent
with `name` in `c::CompositeComponent`.  `c[nothing]` will yield `c`, and `c[x1 => (x2 => (... => (x_n)))]`
will yield `c[x1][x2][...][x_n]`.  Thus, a subcomponent name may be `nothing`, a top-level name,
or, to refer to a nested subcomponent, a nested name implemented as a `Pair`-based linked list.
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
struct CompositeComponent <: Component #{InID, OutID, SC} <: Component
    input::CompositeValue #{InID}
    output::CompositeValue #{OutID}
    subcomponents #::SC # `Tuple` or `NamedTuple` of subcomponents
    node_to_idx::Dict{NodeName, UInt} # maps from `NodeName` to index of this node in the graph
    idx_to_node::Vector{NodeName} # maps index of a node in the graph to its `NodeName`
    graph::SimpleDiGraph # graph of connections between input/output values
    abstract::Union{Nothing, Component} # the component which was `implement`ed to obtain this `CompositeComponent`
    
    # ensure `subcomponents` is either a Tuple or NamedTuple of Components
    # CompositeComponent(i::CompositeValue{I}, o::CompositeValue{O}, sc::T, ni, in, g, abs=nothing) where {I, O, T <: tup_or_namedtup(Component)} =
    #     new{I, O, basetype(T)}(i, o, sc, ni, in, g, abs)

    # CompositeComponent(i::CompositeValue{I}, o::CompositeValue{O}, sc::T, args...) where {I, O, T <: Tuple{Vararg{<:Component}}} =
    #     new{I, O, Tuple}(i, o, sc, args...)
    # CompositeComponent(i::CompositeValue{I}, o::CompositeValue{O}, sc::NT, args...) where {I, O, NT <: NamedTuple{<:Any, <:Tuple{Vararg{<:Component}}}} =
    #     new{I, O, NamedTuple}(i, o, sc, args...)
end
CompositeComponent(i, o, sc, ni, in, g) = new(i, o, sc, ni, in, g, nothing)

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
        (Input(k) for k in keys_deep(input)), (Output(k) for k in keys_deep(output)),
        (CompIn(compname, inname) for (compname, subcomp) in pairs(subcomponents) for inname in keys_deep(inputs(subcomp))),
        (CompOut(compname, outname) for (compname, subcomp) in pairs(subcomponents) for outname in keys_deep(outputs(subcomp)))
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

Base.getindex(c::CompositeComponent, ::Nothing) = c
Base.getindex(c::CompositeComponent, p::Pair) = c[p.first][p.second]
Base.getindex(c::CompositeComponent, name) = c.subcomponents[name]

"""
    get_edges(c::CompositeComponent)

Iterator over edges in the composite component graph.
"""
get_edges(c::CompositeComponent) = (
        c.idx_to_node[src(edge)] => c.idx_to_node[dst(edge)]
        for edge in edges(c.graph)
    )

is_implementation_for(c::CompositeComponent, t::Target) =
    is_implementation_for(inputs(c), t) && is_implementation_for(outputs(c), t) && all(
        is_implementation_for(subc, t) for subc in values(c.subcomponents)
    )

"""
    implement(c::CompositeComponent, t::Target,
        subcomponent_filter::Function = (n -> true),
        input_filter::Function = (n -> true),
        output_filter::Function = (n -> true);
        deep=false
    )

Make progress implementing `c` for `t`.  Implement each subcomponent
whose name passes `subcomponent_filter`, each input value whose name
passes `input_filter`, and each output whose name passes `output_filter`.

If `deep` is false, `implement` will be used for recursive calls (shallow one-step implement);
if `deep` is true, `implement_deep` will be used (fully implementing `c` for `t`).
"""
implement(c::CompositeComponent, t::Target,
    subcomponent_filter::Function = (n -> true),
    input_filter::Function = (n -> true),
    output_filter::Function = (n -> true);
    deep=false
) =
    CompositeComponent(
        implement(inputs(c), t, input_filter; deep),
        implement(outputs(c), t, output_filter; deep),
        map(names(c.subcomponents), c.subcomponents) do name, subcomp
            if subcomponent_filter(name)
                if deep
                    implement_deep(subcomp, t)
                else
                    implement(subcomp, t)
                end
            else
                subcomp
            end
        end,
        c.node_to_idx, c.idx_to_node, c.graph, c
    )

"""
    implement(c::CompositeComponent, t::Target, subcomp_names...)
    implement_deep(c, t::Target, subcomp_names...)

Implement or deep-implement the subcomponents of `c` with names in `subcomp_names`.
This will not implement the input/output values at all; if that is needed, use
the more general `implement(::CompositeComponent, ...)` with name filtering.
"""
implement(c::CompositeComponent, t::Target, subcomp_names...; kwargs...) =
    implement(c, t, in(subcomp_names); kwargs...)

implement_deep(c, t::Target, subcomp_names...; kwargs...) =
    implement(c, t, subcomp_names...; deep=true)

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
    receivers(c::CompositeComponent: name::NodeName)

Iterator over all the `NodeName`s which receive output from the node named `name`.

(If the component is valid, each element of the outputted iterator
will either be an `Input`, `Output` or `CompIn`.)
"""
function receivers(c::CompositeComponent, name::CompIn)
    r = _receivers(c, name)
    @assert isempty(r)
    r
end
receivers(c::CompositeComponent, name::Union{Input, Output, CompOut}) = _receivers(c, name)

_receivers(c::CompositeComponent, name::NodeName) = (
        c.idx_to_node[idx] for idx in outneighbors(c.graph, c.node_to_idx[name])
    )

# TODO: better documentation
# TODO: could I directly have this be a subtype of `CompositeComponent`
# which satisfies some standard interface, rather than it needing to be `implement`ed?
"""
    ComponentGroup <: GenericComponent
    ComponentGroup(subcomponents)

A group of several components, with names given by indices if `subcomponents` is a `Tuple`
and keys if `subcomponents` is a `NamedTuple`.
"""
struct ComponentGroup{T} <: GenericComponent
    subcomponents::T
end
inputs(c::ComponentGroup) = CompositeValue(map(inputs, c.subcomponents))
outputs(c::ComponentGroup) = CompositeValue(map(outputs, c.subcomponents))
implement(c::ComponentGroup, ::Target) =
    CompositeComponent(inputs(c), outputs(c), c.subcomponents, Iterators.flatten((
        (
            Input(i => inname) => CompIn(i, inname)
            for (i, sc) in pairs(c.subcomponents) for inname in keys_deep(inputs(sc))
        ),
        (
            CompOut(i, outname) => Output(i => outname)
            for (i, sc) in pairs(c.subcomponents) for outname in keys_deep(outputs(sc))
        )
    )), c)

"""
    IndexedComponentGroup(subcomponents)

A `ComponentGroup` with subcomponents named via indices of values in `subcomponents`.
"""
IndexedComponentGroup(t) = ComponentGroup(Tuple(t))

"""
    NamedComponentGroup(subcomponents)

A `ComponentGroup` with subcomponent named via keys given keys, given a `subcomponents`
iterator over `(key::Symbol, subcomponent)` pairs.
"""
NamedComponentGroup(n) = ComponentGroup(NamedTuple(n))

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