"""
Var(p̂/p) = (1 - p)/C
where C = AssemblySize × E[Latency] × MaxRate
So:
(1 - p) = AssemblySize × E[Latency] × MaxRate × Var(p̂/p)
"""

using GLMakie

p_fixed = 0.1; p_range = 0.0:0.001:1.0
A_fixed = 10; A_Range = 1:500
L_fixed = 10.; L_range=1.:1.:500.
MR = 0.1
FV_fixed = .1; FV_Range = 0.001:0.001:10.

f = Figure()
maxrate = 0.1
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "Expected Scoring Latency (ms)"
ax.ylabel = "Assembly Size Needed for Scoring P=$p_fixed at Fractional Variance $FV_fixed"
lines!(ax, L_range, [
    ceil((1 - p_fixed) / (l * MR * FV_fixed))
    for l in L_range
])

f


f = Figure()
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "Assembly Size"
ax.ylabel = "Latency Needed for Scoring P=$p_fixed at Fractional Variance $FV_fixed"
lines!(ax, A_Range, [
    (1 - p_fixed) / (a * MR * FV_fixed)
    for a in A_Range
])

f


f = Figure()
maxrate = 0.1
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "p value to score"
ax.ylabel = "Var(p̂/p)"
lines!(ax, p_range, [
    (1 - p) / (L_fixed * A_fixed * MR)
    for p in p_range
])

f


f = Figure()
maxrate = 0.1
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "Expected Latency (ms)"
ax.ylabel = "Var(p̂/p) for p = $p_fixed"
lines!(ax, L_range, [
    (1 - l) / (l * A_fixed * MR * (1 - p_fixed))
    for l in L_range
])

f

### 1/Q Scoring Tradeoffs ###
"""
Var(q̂⁻¹ / q⁻¹) = (1 - q)/(AssemblySize * Latency * MinProb * MaxRate * q)
"""
MP = 0.1
C(a, l) = MR * a * l * MP |> Int
Q_fixed = 0.2
f = Figure()
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "Expected Latency (ms)"
ax.ylabel = "Var(q̂⁻¹/q⁻¹) for q = $Q_fixed, AssemblySize=100, MinProb=$MP"
lines!(ax, L_range, [
    (1 - Q_fixed) / (C(100, l) * Q_fixed)
    for l in L_range
])
f


f = Figure()
ax = Axis(f[1, 1], aspect=1)
Q_range = MP:0.001:1.0
ax.xlabel = "q"
ax.ylabel = "Var(q̂⁻¹/q⁻¹) for AssemblySize=$A_fixed, E[Latency]=$L_fixed (MinProb=$MP)"
lines!(ax, Q_range, [
    (1 - q) / (C(A_fixed, L_fixed) * q)
    for q in Q_range
])
f


### Fractional Variance of Product ###
"""
Var(prod est / prod) = (1 + Var(p̂/p))^n_vars - 1
"""
f = Figure()
ax = Axis(f[1, 1], aspect=1)
ax.xlabel = "Number of variables to score, with each Var(p̂/p) = $FV_fixed"
ax.ylabel = "Fractional variance of estimate of the product of p values"
lines!(ax, 1:20, [
    (1 + FV_fixed)^n - 1
    for n=1:20
])
f