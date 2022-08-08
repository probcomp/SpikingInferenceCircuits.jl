
"""
SimpleProductScoreLine

Spiketrain within a simple (non-auto-normalized) product of dense values.

`spiketrain_factors` is a vector specifying the dense values which are factors for this spiketrain.
Each element of `factors` is a pair `(addr, recip_score::Bool)` giving the address
of the variable which was scored in this factor, and whether the factor is for the direct
or reciprocal probability estimate taken for that value.

`other_factors` is a vector of other values to multiply into the score, assumed to be ready
as soon as all the spiketrain factors are.
"""
struct SimpleProductScoreLine <: SpiketrainSpec
spiketrain_factors::Vector{Tuple{Symbol, Bool}}
other_factors::Vector{<:Real}
line_to_show::SpikelineInScore
end
get_line(spec, tr, trains, productlines; nest_all_at) = productlines[spec]

function sample_product_data(specs, tr, spiketrain_data, spiketrain_data_args)
product_lines = Dict()
for spec in filter(s -> s isa SimpleProductScoreLine, specs)
    product_lines[spec] = sample_product_line(spec, spiketrain_data, spiketrain_data_args)
end
end
function sample_product_line(spec, spiketrain_data, spiketrain_data_args)

end