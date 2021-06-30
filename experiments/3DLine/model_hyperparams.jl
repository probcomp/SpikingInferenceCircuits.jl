# Xs() = 1:10
# Ys() = -5:5
# Heights() = 1:10
ydiv = 1
Vels() = 1:ydiv:3
Xs() = 1:20
Ys() = -5:ydiv:5
Heights() = 1:20
Rs() = Int64(floor(sqrt(Xs()[1]^2 + Heights()[1]^2))):Int64(ceil(norm_3d(Xs()[end], Ys()[end], Heights()[end])))
ϕstep() = 0.1
θstep() = 0.1

# x = 1, y = 5 takes up .19 radians that are unavailable in az.
# vis field is 1.57 radians. 

ϕs() = -1.4:ϕstep():1.4
θs() = -1.4:θstep():1.4
MinProb() = 0.1
