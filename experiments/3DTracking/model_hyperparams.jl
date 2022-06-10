# Xs() = 1:10
# Ys() = -5:5
# Heights() = 1:10
veldiv = 1
ydiv = 1
Vels() = -1:ydiv:1
#Vels() = -1:veldiv:1
PredatorVelScale() = 2
PreyVelScale() = 1
Xs() = 1:20
Ys() = -10:ydiv:10
Zs() = 1:10
X_init = 3
Y_init = 0
Z_init = 5
Rs() = Int64(floor(sqrt(Xs()[1]^2 + Zs()[1]^2))):Int64(ceil(norm_3d(Xs()[end], Ys()[end], Zs()[end])))
ϕstep() = 0.1
θstep() = 0.1
#SphericalVels() = -1.6:θstep():1.6
SphericalVels() = -.5:θstep():.5

# x = 1, y = 5 takes up .19 radians that are unavailable in az.
# vis field is 1.57 radians. 

ϕs() = 0:ϕstep():1.4
θs() = -1.4:θstep():1.4
MinProb() = 0.01

scale_velocity(vel, is_prey) = is_prey ? Int(round(vel / PreyVelScale())) : Int(round(vel / PredatorVelScale()))

