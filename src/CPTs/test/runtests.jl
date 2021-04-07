using Test
using CPTs
using Gen: logpdf

cpt = CPT([[0.5, 0.5], [0.1, 0.9]])

lCPT = LabeledCPT{Bool}(
    [[true, false]],
    [true, false],
    ((x,),) -> x ? [.5, .5] : [.1, .9]
)

@test cpt(1) in (1, 2)
@test lCPT(true) in (true, false)

@test logpdf(cpt, 1, 1) ≈ log(0.5)
@test logpdf(cpt, 2, 1) ≈ log(0.5)
@test logpdf(cpt, 1, 2) ≈ log(0.1)
@test logpdf(cpt, 2, 2) ≈ log(0.9)

@test logpdf(lCPT, true, true) ≈ log(0.5)
@test logpdf(lCPT, false, true) ≈ log(0.5)
@test logpdf(lCPT, true, false) ≈ log(0.1)
@test logpdf(lCPT, false, false) ≈ log(0.9)