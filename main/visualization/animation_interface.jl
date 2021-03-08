"""
    animation_to_frontend_format(animation)

Converts the animation frame list from the simulator (a list of
`(time, (component_name, output_name))` to the JSON format understood by the
front-end.)
"""
animation_to_frontend_format(animation) = Dict("frames" => [
    Dict("time" => time, "group" => convert_compname(compname), "out" => outname)
    for (time, (compname, outname)) in animation
])

convert_compname(::Nothing) = []
convert_compname(name) = [name]
convert_compname((first, rest)::Pair) = prepend!(convert_compname(rest), first)