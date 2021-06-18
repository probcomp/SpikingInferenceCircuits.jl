### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 05d22ac0-cfb2-11eb-0573-a73c27695cce
begin
	using JSServe
	Page()
end

# ╔═╡ 47329ce2-3c79-454c-9ed0-47a95d2dc5c2
RES = 400

# ╔═╡ 911d420d-7571-4487-9f0a-33b2b8c48f85
begin
	using PlutoUI
	using WGLMakie
	set_theme!(resolution=(RES, RES), colormap=:grays, fontsize=12)
	using Gen
	using Distributions
end

# ╔═╡ 46dcdb1d-3527-4149-9b8f-19064e98d022
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 43a36d34-5380-4c83-8b51-7e24efa7ef79
M = ingredients("vanilla_gen_smc.jl")

# ╔═╡ d7d270ef-fa6f-4b9e-9ef1-8b1a7615b758
tr = simulate(M.step_model, (2, 2, 10, -3))

# ╔═╡ 9d61975e-5db5-44da-8e75-d9ee16912b2c
obs_matrix_for_tr(tr) = [x == tr[:obsx] && y == tr[:obsy] for x in M.Positions(), y in M.Positions()]

# ╔═╡ d59b932f-1349-44dd-a313-755a5b6b890c
heatmap(obs_matrix_for_tr(tr))

# ╔═╡ 6c62b5b4-597f-431c-8708-dc970cb5281a
gt_matrix_for_tr(tr) = [x == tr[:xₜ] && y == tr[:yₜ] for x in M.Positions(), y in M.Positions()]

# ╔═╡ ec610508-f605-4839-a173-00a995f80937
function simulate_multiple_stpes(initial_latents, n_steps)
	trs = []
	latents = initial_latents
	for _=1:n_steps
		tr = simulate(M.step_model, latents)
		push!(trs, tr)
		latents = (tr[:xₜ], tr[:vxₜ], tr[:yₜ], tr[:vyₜ])
	end
	return trs
end

# ╔═╡ 1f9ce7b4-7c20-4fc2-96bb-62a90ac079e2
NSTEPS = 10

# ╔═╡ b12da6d7-0b8a-439d-8199-8f67fa8ad123
INIT_LATENTS = (2, 2, 19, -2)

# ╔═╡ c5999ff6-247d-4fd1-a60b-b853657e16aa
trs = simulate_multiple_stpes(INIT_LATENTS, NSTEPS)

# ╔═╡ bea89f5c-2ccb-42f4-bfcd-9a8d317dac29
t__ = let foo = trs
	Observable(1)
end

# ╔═╡ 5a9030d6-1fc9-48a3-93b8-03d48811d1f9
@bind t PlutoUI.Slider(1:NSTEPS)

# ╔═╡ 83b284a4-64d0-4bd9-94a3-3c9bed7be14a
t__[] = let foo = trs
	t
end

# ╔═╡ ad0503b9-7a75-48a8-8401-c3720895f082


# ╔═╡ da5916b6-0fd2-42a8-a87d-fb18374c8761
both_matrix_for_tr(tr) = 2 * obs_matrix_for_tr(tr) + gt_matrix_for_tr(tr)

# ╔═╡ f98c4b47-30e9-4be1-8307-f822ab04c07f
md"""
White is observation; grey is ground truth.
"""

# ╔═╡ 2c969200-0ff2-4693-a53e-e5f72d6c0e83
heatmap(
	WGLMakie.@lift(both_matrix_for_tr(trs[$t__]))
	)

# ╔═╡ 45b250bf-6ace-4c69-9e09-c0cd37d3928f


# ╔═╡ bcfe53c2-e94c-45de-b7d6-16b5cc317443
md"""
### Testing SMC
"""

# ╔═╡ 7818f934-30aa-4faf-a4ec-f25f955e2e53
obs_chs = [
		choicemap(
			(:obsx, trs[t][:obsx]),
			(:obsy, trs[t][:obsy])
		)
		for t=1:NSTEPS
	]

# ╔═╡ 6243df45-d621-45e4-afc4-9ba33a5025d9
(weighted_samples, unweighted_samples) = M.particle_filter(
	obs_chs, INIT_LATENTS, 20
)

# ╔═╡ f6de1e8b-45da-456c-be23-851e41dcc80d
slider = JSServe.Slider(1:NSTEPS)

# ╔═╡ 0d1e59cd-5000-46df-bdb4-c5dd30ceef21
inferred_vs_true_img(t, time_to_inferred_ch) =
	3 * obs_matrix_for_tr(trs[t]) +
	2 * gt_matrix_for_tr(trs[t]) + 
		gt_matrix_for_tr(time_to_inferred_ch(t))

# ╔═╡ 7bd3b5cd-b499-45bd-a7fa-3a451ff45eb8
heatmap(WGLMakie.@lift(inferred_vs_true_img($slider, t -> unweighted_samples[t][1])))

# ╔═╡ f34ece46-a511-4248-ae91-714162469260


# ╔═╡ 4fc5720b-23b6-4d9b-9f50-ae9a6fa5919e
dumb_inferences = M.dumb_select_latents(obs_chs, M.tuple_to_latent_choicemap(INIT_LATENTS))

# ╔═╡ bf4115e8-a368-4cf3-98cd-66ee225dc736
heatmap(WGLMakie.@lift(inferred_vs_true_img($slider, t -> dumb_inferences[t])))

# ╔═╡ a9ab8f55-3ee5-4256-a14a-13f2c6642968


# ╔═╡ 46d859da-45d8-443d-a722-63f25748cb6a


# ╔═╡ bbde5ff6-491e-403d-9988-221f868d21ad
M.joint_trace_logprob(obs_chs, [unweighted_samples[t][1] for t=3:4])

# ╔═╡ 7bf5e01e-ff77-444e-acd8-c8ed3ef4f280


# ╔═╡ Cell order:
# ╟─05d22ac0-cfb2-11eb-0573-a73c27695cce
# ╠═47329ce2-3c79-454c-9ed0-47a95d2dc5c2
# ╟─911d420d-7571-4487-9f0a-33b2b8c48f85
# ╟─46dcdb1d-3527-4149-9b8f-19064e98d022
# ╠═43a36d34-5380-4c83-8b51-7e24efa7ef79
# ╠═d7d270ef-fa6f-4b9e-9ef1-8b1a7615b758
# ╠═d59b932f-1349-44dd-a313-755a5b6b890c
# ╟─9d61975e-5db5-44da-8e75-d9ee16912b2c
# ╟─6c62b5b4-597f-431c-8708-dc970cb5281a
# ╟─ec610508-f605-4839-a173-00a995f80937
# ╠═1f9ce7b4-7c20-4fc2-96bb-62a90ac079e2
# ╠═b12da6d7-0b8a-439d-8199-8f67fa8ad123
# ╠═c5999ff6-247d-4fd1-a60b-b853657e16aa
# ╟─bea89f5c-2ccb-42f4-bfcd-9a8d317dac29
# ╟─5a9030d6-1fc9-48a3-93b8-03d48811d1f9
# ╟─83b284a4-64d0-4bd9-94a3-3c9bed7be14a
# ╠═ad0503b9-7a75-48a8-8401-c3720895f082
# ╠═da5916b6-0fd2-42a8-a87d-fb18374c8761
# ╟─f98c4b47-30e9-4be1-8307-f822ab04c07f
# ╠═2c969200-0ff2-4693-a53e-e5f72d6c0e83
# ╠═45b250bf-6ace-4c69-9e09-c0cd37d3928f
# ╟─bcfe53c2-e94c-45de-b7d6-16b5cc317443
# ╟─7818f934-30aa-4faf-a4ec-f25f955e2e53
# ╠═6243df45-d621-45e4-afc4-9ba33a5025d9
# ╠═f6de1e8b-45da-456c-be23-851e41dcc80d
# ╠═0d1e59cd-5000-46df-bdb4-c5dd30ceef21
# ╠═7bd3b5cd-b499-45bd-a7fa-3a451ff45eb8
# ╠═f34ece46-a511-4248-ae91-714162469260
# ╟─4fc5720b-23b6-4d9b-9f50-ae9a6fa5919e
# ╠═bf4115e8-a368-4cf3-98cd-66ee225dc736
# ╠═a9ab8f55-3ee5-4256-a14a-13f2c6642968
# ╠═46d859da-45d8-443d-a722-63f25748cb6a
# ╠═bbde5ff6-491e-403d-9988-221f868d21ad
# ╠═7bf5e01e-ff77-444e-acd8-c8ed3ef4f280
