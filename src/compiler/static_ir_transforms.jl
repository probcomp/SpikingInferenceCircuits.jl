module StaticIRTransforms
using Gen
using Bijections
using Gen: StaticIRGenerativeFunction
using Gen: StaticIRNode, RandomChoiceNode, JuliaNode, ArgumentNode, GenerativeFunctionCallNode
using Gen: Bernoulli, UniformDiscrete, Categorical

import ..CPT, ..LabeledCPT

export to_labeled_cpts

# Note: this currently only works when every input to every discrete distribution
# should be removed from the graph!  This is common when setting parameters to
# bernoulli, categorical, etc., but may be incorrect in other cases!
function to_labeled_cpts(ir::StaticIR, arg_domains)
    name_to_domain = get_domains(ir.nodes, arg_domains)
    names_to_delete = Set{Symbol}()
    to_replace = Dict{Symbol, StaticIRNode}()

    for node in ir.choice_nodes
        if !is_cpt(node.dist)
            to_replace[node.name] = get_labeled_cpt_node(node, name_to_domain)
            for parent in node.inputs
                push!(names_to_delete, parent.name)
            end
        end
    end
    for node in ir.call_nodes
        to_replace[node.name] = GenerativeFunctionCallNode(
            to_labeled_cpts(
                node.generative_function,
                Tuple(x -> name_to_domain[x.name] for x in node.inputs)
            ),
            node.inputs, node.addr, node.name, node.typ
        )
    end

    builder = StaticIRBuilder()
    for node in ir.nodes
        if !(node.name in names_to_delete)
            new_node = haskey(to_replace, node.name) ? to_replace[node.name] : node
            Gen._add_node!(builder, new_node)
            if ir.return_node.name == new_node.name
                Gen.set_return_node!(builder, new_node)
            end
        end
    end

    return build_ir(builder)
end
to_labeled_cpts(gf::StaticIRGenerativeFunction, arg_domains) = to_labeled_cpts(Gen.get_ir(typeof(gf)), arg_domains)

is_cpt(::CPT) = true
is_cpt(::LabeledCPT) = true
is_cpt(::Gen.Distribution) = false

function get_labeled_cpt_node(node::RandomChoiceNode, name_to_domain)
    @assert all(n isa JuliaNode for n in node.inputs) "Assumptions about type of IR violated for node $(node.name)!"
    parent_inputs = collect(Iterators.flatten(pa.inputs for pa in node.inputs))
    parent_input_domains = Tuple(name_to_domain[x.name] for x in parent_inputs)
    input_domains = Tuple(name_to_domain[x.name] for x in node.inputs)

    cpt = get_labeled_cpt(
        node.dist, map(n -> n.fn, node.inputs),
        parent_input_domains, input_domains
    )
    return RandomChoiceNode(cpt, parent_inputs, node.addr, node.name, node.typ)
end

function _labeled_cpt(rettype, parent_domains, output_domain, parent_fns, fns_out_to_prob)
    assmt_to_val(assmt) = fns_out_to_prob(Tuple(fn(v) for (fn, v) in zip(parent_fns, assmt)))
    LabeledCPT{rettype}(collect(parent_domains), collect(output_domain), assmt_to_val)
end
get_labeled_cpt(d::Distribution, parent_fns, parent_domains, input_domains) =
    _labeled_cpt(
        Gen.get_return_type(d),
        parent_domains,
        get_domain(d, input_domains),
        parent_fns,
        assmt_to_probs(d)
    )


function get_domains(nodes, arg_domains)
    name_to_domain = Dict{Symbol, Vector}()

    # insert argument domains
    for (node, domain) in zip(nodes, arg_domains)
        name_to_domain[node.name] = domain
    end

    for node in nodes[(length(arg_domains) + 1):end]
        handle_node!(node, name_to_domain)
    end

    display(name_to_domain)
    return name_to_domain
end

# handle_node!(::ArgumentNode, _) = nothing
function handle_node!(node::JuliaNode, name_to_domain)
    input_domains = (name_to_domain[parent.name] for parent in node.inputs)
    assmts = Iterators.product(input_domains...)
    name_to_domain[node.name] = [node.fn(assmt...) for assmt in assmts]
end
function handle_node!(node::RandomChoiceNode, name_to_domain)
    domain = get_domain(node.dist, [name_to_domain[x.name] for x in node.inputs])
    name_to_domain[node.name] = domain
end

## distribution specific: ##

get_domain(::Bernoulli, _) = [true, false]
assmt_to_probs(::Bernoulli) = ((p,),) -> [p, 1 - p]

get_domain(::Categorical, arg_domains) = 1:length(only(arg_domains))
assmt_to_probs(::Categorical) = ((pvec,),) -> pvec

get_domain(::UniformDiscrete, (start_dom, end_dom)) = minimum(start_dom):maximum(end_dom)
assmt_to_probs(::UniformDiscrete) = ((min, max),) -> [1/(max - min) for _=min:max]

# get_labeled_cpt(::Bernoulli, parent_fns, parent_domains) =
#     _labeled_cpt(
#         Bool,
#         parent_domains,
#         [true, false],
#         parent_fns,
#         ((p,),) -> [p, 1-p]
#     )

# function get_labeled_cpt(::Categorical, parent_fns, parent_domains)
#     @assert length(parent_fns) == 1
#     @assert length(parent_domains == 1)
#     out_length = length(first(parent_fns)(first(first(parent_domains))))

#     _labeled_cpt(
#         Int,
#         parent_domains,
#         parent_fns,
#         1:out_length,
#         ((v,),) -> v
#     )
# end
end