Vels() = 2:4
Xs() = 1:10
Ys() = -5:5
Heights() = 1:10
Rs() = 0:Int64(ceil(norm_3d(Xs()[end], Ys()[end], Heights()[end])))
ϕs() = -1.4:ϕstep():1.4
θs() = -1.4:θstep():1.4
ϕstep() = 0.2
θstep() = 0.2

MinProb() = 0.1