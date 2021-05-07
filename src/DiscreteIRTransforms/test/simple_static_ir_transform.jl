@gen (static) function foo(x::Bool)
    a = !x
    y ~ bernoulli([0.0, 1.0][a ? 1 : 2])
    return y
end

input_domains = [EnumeratedDomain([true, false])]

with_lcpts = to_labeled_cpts(foo, input_domains)

(with_cpts, bijs) = to_indexed_cpts(foo, input_domains)
@load_generated_functions()

og_domains = get_domains(get_ir(foo).nodes, input_domains)
new_domains = get_domains(get_ir(with_lcpts).nodes, input_domains)

@test all(haskey(og_domains, name) && og_domains[name] == new_dom for (name, new_dom) in new_domains)

@test with_lcpts(true) == true
@test with_lcpts(false) == false
@test with_cpts(1) == 1
@test with_cpts(2) == 2