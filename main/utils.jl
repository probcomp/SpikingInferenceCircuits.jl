tup_or_namedtup(T) = Union{
    Tuple{Vararg{<:T}},
    NamedTuple{<:Any, <:Tuple{Vararg{<:T}}}
}

basetype(T) = T.name.wrapper