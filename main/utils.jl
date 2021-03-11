tup_or_namedtup(T) = Union{
    Tuple{Vararg{<:T}},
    NamedTuple{<:Any, <:Tuple{Vararg{<:T}}}
}

basetype(T) = T.name.wrapper

"""
    zip_with_defaults(firstitr, itrs, defaults)

Similar to `zip(firstitr, itrs...)`, but `itrs[j]` runs out of elements,
`defaults[j]` will be placed in its position instead of the `zip` ending.
(So `firstitr` sets the length.)

### Example
```julia
julia> collect(zip_with_defaults(1:4, (1:2, 1:3), (100, 200)))
# = [(1, 1, 1), (2, 2, 2), (3, 100, 3), (4, 100, 200)]
julia> collect(zip_with_defaults(1:3, (1:2,), (100,)))
# = [(1, 1), (2, 2), (3, 100)]
"""
zip_with_defaults(firstitr, itrs, defaults) = zip(
        firstitr,
        (
            Iterators.flatten((itr, Iterators.repeated(default)))
            for (itr, default) in zip(itrs, defaults)
        )...
    )