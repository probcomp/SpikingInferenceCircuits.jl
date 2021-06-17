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
RES = 200

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
M = ingredients("model_proposal.jl")

# ╔═╡ d7d270ef-fa6f-4b9e-9ef1-8b1a7615b758
tr = simulate(M.step_model, (2, 2, 10, -3))

# ╔═╡ 9cefee69-238f-4ada-a680-0469ef91e835
heatmap([x == 2 && y == 10 for x in M.Positions(), y in M.Positions()])

# ╔═╡ bd8c3325-c84a-4e6e-88c7-d5cd9675255f
heatmap([x == tr[:xₜ] && y == tr[:yₜ] for x in M.Positions(), y in M.Positions()])

# ╔═╡ 9d61975e-5db5-44da-8e75-d9ee16912b2c
obs_matrix_for_tr(tr) = [x == tr[:obsx] && y == tr[:obsy] for x in M.Positions(), y in M.Positions()]

# ╔═╡ d59b932f-1349-44dd-a313-755a5b6b890c
heatmap(obs_matrix_for_tr(tr))

# ╔═╡ 6c62b5b4-597f-431c-8708-dc970cb5281a
gt_matrix_for_tr(tr) = [x == tr[:xₜ] && y == tr[:yₜ] for x in M.Positions(), y in M.Positions()]

# ╔═╡ 74198fd7-82c1-4c86-9676-93ce8e7de4bd


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

# ╔═╡ c5999ff6-247d-4fd1-a60b-b853657e16aa
trs = simulate_multiple_stpes((2, 2, 19, -2), NSTEPS)

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

# ╔═╡ b4be8ab6-d3b5-4ad4-8bec-06f3740ab61f


# ╔═╡ 45b250bf-6ace-4c69-9e09-c0cd37d3928f


# ╔═╡ cdf41529-944f-41f9-9c81-f167397718ff


# ╔═╡ bcfe53c2-e94c-45de-b7d6-16b5cc317443
md"""
### Testing SMC
"""

# ╔═╡ 6243df45-d621-45e4-afc4-9ba33a5025d9


# ╔═╡ Cell order:
# ╠═05d22ac0-cfb2-11eb-0573-a73c27695cce
# ╠═47329ce2-3c79-454c-9ed0-47a95d2dc5c2
# ╠═911d420d-7571-4487-9f0a-33b2b8c48f85
# ╠═46dcdb1d-3527-4149-9b8f-19064e98d022
# ╠═43a36d34-5380-4c83-8b51-7e24efa7ef79
# ╠═d7d270ef-fa6f-4b9e-9ef1-8b1a7615b758
# ╠═9cefee69-238f-4ada-a680-0469ef91e835
# ╠═bd8c3325-c84a-4e6e-88c7-d5cd9675255f
# ╠═d59b932f-1349-44dd-a313-755a5b6b890c
# ╠═9d61975e-5db5-44da-8e75-d9ee16912b2c
# ╠═6c62b5b4-597f-431c-8708-dc970cb5281a
# ╠═74198fd7-82c1-4c86-9676-93ce8e7de4bd
# ╠═ec610508-f605-4839-a173-00a995f80937
# ╠═1f9ce7b4-7c20-4fc2-96bb-62a90ac079e2
# ╠═c5999ff6-247d-4fd1-a60b-b853657e16aa
# ╠═bea89f5c-2ccb-42f4-bfcd-9a8d317dac29
# ╠═5a9030d6-1fc9-48a3-93b8-03d48811d1f9
# ╟─83b284a4-64d0-4bd9-94a3-3c9bed7be14a
# ╠═ad0503b9-7a75-48a8-8401-c3720895f082
# ╠═da5916b6-0fd2-42a8-a87d-fb18374c8761
# ╟─f98c4b47-30e9-4be1-8307-f822ab04c07f
# ╠═2c969200-0ff2-4693-a53e-e5f72d6c0e83
# ╠═b4be8ab6-d3b5-4ad4-8bec-06f3740ab61f
# ╠═45b250bf-6ace-4c69-9e09-c0cd37d3928f
# ╠═cdf41529-944f-41f9-9c81-f167397718ff
# ╟─bcfe53c2-e94c-45de-b7d6-16b5cc317443
# ╠═6243df45-d621-45e4-afc4-9ba33a5025d9
