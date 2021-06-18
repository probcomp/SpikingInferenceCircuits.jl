using GLMakie
RES = 400
set_theme!(resolution=(RES, RES), colormap=:grays, fontsize=12)

matrix(truex, truey) = [x == truex && y == truey for x in Positions(), y in Positions()]
gt_matrix(ch) = matrix(ch[:latents => :xₜ], ch[:latents => :yₜ])
obs_matrix(ch) = matrix(ch[:obs => :obsx], ch[:obs => :obsy])
both_matrix(ch) = 2 * obs_matrix(ch) + gt_matrix(ch)

# inferred_vs_true_img() =
# 	3 * obs_matrix_for_tr(trs[t]) +
# 	2 * gt_matrix_for_tr(trs[t]) + 
# 		gt_matrix_for_tr(time_to_inferred_ch(t))

# tr = simulate(dm, (10,))
ch(tr, t) = get_submap(
    get_choices(tr),
    t == 0 ? :init : :steps => t
)

t = Observable(1)
show_matrix(_t) = both_matrix(ch(tr, _t)) + 3*gt_matrix(ch(inf, _t))
m = GLMakie.@lift show_matrix($t)

# white = inferred
# light grey = obs
# dark grey = gt