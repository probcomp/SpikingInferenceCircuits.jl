# Xs() = 1:10
# Ys() = -5:5
# Heights() = 1:10
veldiv = 1
ydiv = 1
zdiv = ydiv
Vels() = -2:ydiv:2
#Vels() = -1:veldiv:1
PredatorVelScale() = 2
PreyVelScale() = 1
y_lb = -20
y_ub = 20
z_lb = -20
z_ub = 20
fish_origin = (0, 0, 0)
Xs() = 1:(y_ub * 2)
Ys() = y_lb:ydiv:y_ub
Zs() = z_lb:zdiv:z_ub
YZs() = [(y, z) for y in Ys() for z in Zs()]
X_init = 20
Y_init = y_lb / 4
Z_init = 3
Rs() = Int64(floor(sqrt(Xs()[1]^2))):Int64(ceil(norm_3d(Xs()[end], Ys()[end], Zs()[end])))
ϕstep() = 0.1
θstep() = 0.1
#SphericalVels() = -1.6:θstep():1.6
SphericalVels() = -1.5:θstep():1.5
tanksize = 302

#SphericalVels() = -.5:.1:.5


# x = 1, y = 5 takes up .19 radians that are unavailable in az.
# vis field is 1.57 radians. 

ϕs() = -1.4:ϕstep():1.4
θs() = -1.4:θstep():1.4
#MinProb() = 0.01
#MinProb() = 0.001
ProbEstimates.MinProb() = .001
MinProb() = ProbEstimates.MinProb()

scale_velocity(vel, is_prey) = is_prey ? Int(round(vel / PreyVelScale())) : Int(round(vel / PredatorVelScale()))

