# Xs() = 1:10
# Ys() = -5:5
# Heights() = 1:10
ydiv = 1
#Vels() = -3:ydiv:3
Vels() = -1:ydiv:1
Xs() = 1:8
Ys() = -2:ydiv:2
Zs() = 1:8
X_init = 3
Y_init = 0
Z_init = 5
Rs() = Int64(floor(sqrt(Xs()[1]^2 + Zs()[1]^2))):Int64(ceil(norm_3d(Xs()[end], Ys()[end], Zs()[end])))
ϕstep() = 0.1
θstep() = 0.1

# x = 1, y = 5 takes up .19 radians that are unavailable in az.
# vis field is 1.57 radians. 

ϕs() = 0:ϕstep():1.4
θs() = -1.4:θstep():1.4
MinProb() = 0.0

n_dynamic_states = length(Xs()) * length(Ys()) * length(Zs()) * length(Vels())^3
n_transient_states = length(Rs()) * length(ϕs()) * length(θs())