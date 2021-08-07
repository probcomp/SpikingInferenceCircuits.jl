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

@test get_line(SampledValue(:x), tr) isa String
@test get_line(FwdScoreText(:x), tr) isa String
@test get_line(RecipScoreText(:x), tr) isa String


###
propose_sampling_tree = Dict(:x => [], :y => [:x, :z], :z => [:x])
assess_sampling_tree = Dict(:x => [], :y => [:x], :z => [:x, :y])
propose_addr_topological_order = [:x, :z, :y]

linespecs = [
    ScoreLine(true, :x, CountAssembly()),
    ScoreLine(true, :y, CountAssembly()),
    ScoreLine(true, :z, CountAssembly()),
    ScoreLine(false, :x, CountAssembly()),
    ScoreLine(false, :y, CountAssembly()),
    ScoreLine(false, :z, CountAssembly()),
    VarValLine(:x, 1), VarValLine(:x, 2),
    VarValLine(:y, 1), VarValLine(:y, 2),
    VarValLine(:z, 1), VarValLine(:z, 2)
]

lines = get_lines(linespecs, tr,
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


### Test visualization ###
# lines_for_addr(a) = [
#     VarValLine(a, 1), VarValLine(a, 2), SampledValue(a),    
    
#     # RecipScoreLine(a, CountAssembly()),
#     [RecipScoreLine(a, NeuronInCountAssembly(i)) for i=1:5]...,
#     RecipScoreLine(a, IndLine()),
#     RecipScoreText(a),

#     [FwdScoreLine(a, NeuronInCountAssembly(i)) for i=1:5]...,
#     FwdScoreLine(a, IndLine()),
#     FwdScoreText(a),
# ]
# linespecs = [
#    lines_for_addr(:x)...,
#    lines_for_addr(:y)...,
#    lines_for_addr(:z)...
# ]
# lines = get_lines(linespecs, tr,
#     (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order)
# )
# labels = get_labels(linespecs)
# SpiketrainViz.draw_spiketrain_figure(lines; labels, xmin=0, resolution=(1280, 1000))


groups_for_addr(a) = [
    LabeledLineGroup(SampledValue(a), [VarValLine(a, 1), VarValLine(a, 2)]),
    LabeledLineGroup(RecipScoreText(a), [
        [RecipScoreLine(a, NeuronInCountAssembly(i)) for i=1:5]...,
        RecipScoreLine(a, IndLine())
    ]),
    LabeledLineGroup(FwdScoreText(a), [
        [FwdScoreLine(a, NeuronInCountAssembly(i)) for i=1:5]...,
        FwdScoreLine(a, IndLine())
    ]),
]

linegroups = [
    groups_for_addr(:x)...,
    groups_for_addr(:y)...,
    groups_for_addr(:z)...
]

lines = get_lines(linegroups, tr,
    (propose_sampling_tree, assess_sampling_tree, propose_addr_topological_order)
)
labels = get_labels(linegroups)
grouplabels = get_group_labels(linegroups, tr)

f = SpiketrainViz.draw_spiketrain_figure(lines; labels, grouplabels, xmin=0, resolution=(1280, 1000));