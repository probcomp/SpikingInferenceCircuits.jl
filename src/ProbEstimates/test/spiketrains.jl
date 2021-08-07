using Test
using Gen, ProbEstimates
ProbEstimates.use_noisy_weights!()

@gen function model()
    x ~ Cat([0.9, 0.1])
    y ~ Cat(x == 1 ? [0.1, 0.9] : [0.9, 0.1])
    z ~ Cat(x == y ? [0.1, 0.9] : [0.9, 0.1])
end

@gen function proposal()
    x ~ Cat([0.1, 0.9])
    z ~ Cat(x == 1 ? [0.1, 0.9] : [0.9, 0.1])
    y ~ Cat(x == z ? [0.1, 0.9] : [0.9, 0.1])
end


### Tests for storing score, recip score: ###
choices, propscore, _ = propose(proposal, ())
tr, _ = generate(model, (), choices)
@test choices[:x => :recip_score] == get_choices(tr)[:x => :recip_score]
@test !isnothing(get_choices(tr)[:x => :fwd_score])

choices2, propscore, _ = propose(proposal, ())
new_tr, wt = update(tr, choices2);
@test choices2[:x => :recip_score] == get_choices(new_tr)[:x => :recip_score]
@test get_choices(tr)[:x => :fwd_score] != get_choices(new_tr)[:x => :fwd_score]

### Tests for text lines ###
using ProbEstimates.Spiketrains

@test get_line(SampledValue(:x), tr) == "x=$(tr[:x])"
@test get_line(FwdScoreText(:x), tr) isa String
@test get_line(RecipScoreText(:x), tr) isa String


###
propose_sampling_tree = Dict(:x => [], :y => [:x, :z], :z => [:x])
assess_sampling_tree = Dict(:x => [], :y => [:x], :z => [:x, :y])
propose_addr_topological_order = [:x, :z, :y]
lines = get_lines(
    [
        ScoreLine(true, :x, CountAssembly()),
        ScoreLine(true, :y, CountAssembly()),
        ScoreLine(true, :z, CountAssembly()),
        ScoreLine(false, :x, CountAssembly()),
        ScoreLine(false, :y, CountAssembly()),
        ScoreLine(false, :z, CountAssembly()),
        VarValLine(:x, 1), VarValLine(:x, 2),
        VarValLine(:y, 1), VarValLine(:y, 2),
        VarValLine(:z, 1), VarValLine(:z, 2)
    ],
    tr,
    (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order)
)
(xrecip, yrecip, zrecip, xfwd, yfwd, zfwd, xv1, xv2, yv1, yv2, zv1, zv2) = lines
xvaltime = only(vcat(xv1, xv2)); yvaltime = only(vcat(yv1, yv2)); zvaltime = only(vcat(zv1, zv2))
@test xvaltime < zvaltime < yvaltime # samples in topological order
@test first(xrecip) > xvaltime
@test first(yrecip) > yvaltime
@test first(zrecip) > zvaltime
@test first(xfwd) > xvaltime
@test first(yfwd) > max(xvaltime, yvaltime, zvaltime)
@test first(zfwd) > max(xvaltime, yvaltime, zvaltime)