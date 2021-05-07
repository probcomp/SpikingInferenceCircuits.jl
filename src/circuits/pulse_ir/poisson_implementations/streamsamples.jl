struct PoissonStreamSamples <: ConcretePulseIRPrimitive
    P::Matrix{Float64}
    ΔT::Float64
    overall_on_rate::Float64
    overall_off_rate::Float64
end
Circuits.abstract(ss::PoissonStreamSamples) = ConcreteStreamSamples(
    ss.P, ss.ΔT, T -> Distributions.Poisson(ss.overall_on_rate * T)
)
Circuits.target(ss::PoissonStreamSamples) = target(abstract(ss))
Circuits.inputs(ss::PoissonStreamSamples) = inputs(abstract(ss))
Circuits.outputs(ss::PoissonStreamSamples) = outputs(abstract(ss))

out_domain_size(p::PoissonStreamSamples) = out_domain_size(abstract(abstract(p)))
in_domain_size(p::PoissonStreamSamples) = in_domain_size(abstract(abstract(p)))
prob_output_given_input(p::PoissonStreamSamples, outval) = prob_output_given_input(abstract(abstract(p)), outval)

### check whether a function looks like
### T -> Distributions.Poisson(T * rate)
### or T -> Distribuitons.Poisson(rate * T)
function expected_lines_field()
    rate = 1 + 5
    f = (x -> Distributions.Poisson(x * rate))
    return Base.uncompressed_ast(methods(f).ms[1]).code
end
function expected_lines_const()
    f = (x -> Distributions.Poisson(x * 50))
    return Base.uncompressed_ast(methods(f).ms[1]).code
end
is_valid_multline(line) = (
    line isa Expr && line.head == :call &&
    eval(line.args[1]) == (*) &&
    (
        line.args[2] == Core.SlotNumber(2) ||
        line.args[3] == Core.SlotNumber(2)
    ) && (line.args[2] != line.args[3])
)
is_poisson_lines(lines) = (
        (try
            eval(lines[1]) in map(eval, (expected_lines_field()[1], expected_lines_const()[1]))
        catch e
            false
        end) &&
        lines[end-1] in (expected_lines_field()[end-1], expected_lines_const()[end-1]) &&
        lines[end] in (expected_lines_field()[end], expected_lines_const()[end]) &&
        is_valid_multline(lines[end-2])
    )
is_poisson_fn(f) = is_poisson_lines(Base.uncompressed_ast(methods(f).ms[1]).code)
@assert is_poisson_fn(x -> Distributions.Poisson(x * 50))
@assert is_poisson_fn(x -> Distributions.Poisson(50 * x))
@assert !is_poisson_fn(x -> Distributions.Poisson(x*x))
@assert !is_poisson_fn(x -> x*50)
# TODO: move these tests elsewhere

function PoissonStreamSamples(ss::ConcreteStreamSamples, off_rate)
    @assert ss.dist_on_num_samples(1) isa Distributions.Poisson "To have a Poisson implementation, ss.dist_on_num_samples(1) must be a `Distributions.Poisson` but instead it is $(ss.dist_on_num_samples(1))."
    on_rate = ss.dist_on_num_samples(1).λ
    @assert is_poisson_fn(ss.dist_on_num_samples) "To have a Poisson implementation, ss.dist_on_num_samples must look like `x -> Distributions.Poisson(x * rate)`, but instead it looks like $(Base.uncompressed_ast(methods(ss.dist_on_num_samples).ms[1]).code)."
    return PoissonStreamSamples(ss.P, ss.ΔT, on_rate, off_rate)
end

Circuits.implement(p::PoissonStreamSamples, ::Spiking) =
    let bias = log(p.overall_off_rate),
        base_weight = log(p.overall_on_rate) - bias
            CompositeComponent(
                inputs(p), outputs(p),
                Tuple(
                    PoissonNeuron(
                        [
                            x -> min(1, x) × (
                                log(prob_output_given_input(
                                    p, outval
                                )[inval]) + base_weight
                            )
                            for inval=1:in_domain_size(p)
                        ],
                        p.ΔT, u -> exp(u + bias)
                    )
                    for outval = 1:out_domain_size(p)
                ),
                Iterators.flatten((
                    (
                        Input(inval) => CompIn(outval, inval)
                        for outval=1:out_domain_size(p)
                            for inval=1:in_domain_size(p)
                    ),
                    (
                        CompOut(outval, :out) => Output(outval)
                        for outval=1:out_domain_size(p)
                    )
                )),
                p
            )
    end

# The only failure mode is emitting a spike before turned on.
failure_probabability_bound(p::PoissonStreamSamples) = exp(-p.overall_off_rate)