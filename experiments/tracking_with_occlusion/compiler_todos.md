#### Model Compilation

##### High priority:
- Ensure compilation with `Map` combinator works
- Nested `Map` combinator

- Remove arguments to GFs which have domain 1
    - Likewise, remove JuliaNodes with 0 arguments that output the constant
    values with the domains equal to 1

##### Medium priority:
- Handle distributions like `VelCat` where we should construct a `LabeledCPT` by evaluating
function calls to fill in the probability table.
- Handle tuples & tuple return values better
- Handle GFs with no arguments


#### SMC Compilation