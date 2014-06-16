import matplotlib.pyplot as plt
import numpy as np
import sys

#----------------------------------------------------------
# function: calcualte derivatives
def f(r, t, n):
    x = "FILL IN" 
    y = "FILL IN"
    
    fx = "FILL IN"
    fy = "FILL IN"
    return np.array(["FILL IN"])

#----------------------------------------------------------
# function: runge kutta  
def RK4(r, t, h, n):

    xp = []; yp = []; tp = []
    xp.append(r[0]); yp.append(r[1]); tp.append(t)

    # integrate until the density vanishes
    while "FILL IN":

        # runge kutta coefficients
        k1 = h*f(r, t, n)
        k2 = h*f(r+0.5*k1, t+0.5*h, n)
        k3 = h*f(r+0.5*k2, t+0.5*h, n)
        k4 = h*f(r+k3, t+h, n)

        # update the solution 
        r += (k1+2*k2+2*k3+k4)/6.0
        t += h

        # store the solution
        xp.append("FILL IN")
        yp.append("FILL IN")
        tp.append("FILL IN")

    # return the complete solutions as numpy arrays
    return np.array(xp), np.array(yp), np.array(tp)

#----------------------------------------------------------
# main body of the program

# parameters and initial conditions
h = 0.1     # step size
t0 = h      # initial radius

# hold the solutions for eacn index n
theta_sol = []
xi_sol    = []

# solve the problem for different index n's
for n in [0, 1]:

    # boundary conditions
    r0 = np.array(["FILL IN"])

    # 4th order runge kutta integration
    x, y, t = RK4(r0, t0, h, n)

    # store the entire solution for n
    theta_sol.append(x)
    xi_sol.append(t)


# generate true solutions
sol = []
# for n = 0 case
sol.append("FILL IN")
# for n = 1 case
sol.append("FILL IN")

#----------------------------------------------------------
# plotting 

# create image size
plt.figure(figsize=(10,4))

# plot the exact and numerical solution for n = 0
plt.subplot(1,2,1)
plt.plot(xi_sol[0], sol[0], 'r', linewidth=1.5, alpha=0.4 )
plt.plot(xi_sol[0], theta_sol[0], 'ro', alpha=0.7)
plt.xlabel(r"$\xi$", fontsize=20)
plt.ylabel(r"$\theta$", fontsize=20)

# plot the exact and numerical solution for n = 1
plt.plot(xi_sol[1], sol[1], 'b', linewidth=1.5, alpha=0.4)
plt.plot(xi_sol[1], theta_sol[1], 'bo', alpha=0.7)
plt.xlabel(r"$\xi$", fontsize=20)
plt.ylabel(r"$\theta$", fontsize=20)
plt.title("Radial profiles for n=0,1")
plt.xlim(0, 4)
plt.ylim(0, 1)

# add legend
l = plt.legend(["n=0: Eact ", "n=0: Numerical ",
    "n=1: Eact ", "n=1: Numerical "], prop={'size':12})
l.draw_frame(False)

plt.subplot(1,2,2)

# add the value of the background density
# remember we set the density to 1.0
rho = np.append(theta_sol[1], np.array([1.0E-5]))

# create the grid to map our solution
res = 256                            # resolution of the image
x = np.linspace(-5, 5, res)          # equally space points in 1d
X, Y = np.meshgrid(x,x);             # 2d grid from set of 1d points
D = np.sqrt(X**2 + Y**2).flatten()   # matrix of the distance form the
                                     # origin to point X Y

# radius points from the inner boundary to the
# shock
radius = xi_sol[1].copy()       # create radius
dr = np.mean(np.diff(radius))   # bin length

# we have to handle values that fall outside
# of our integration limits in our box
r_bin = radius - 0.5*dr  # left side of the bins

# add the last bin and one more bin to hanlde values outside
# the shock
r_bin = np.append(radius, np.array([radius[-1]+0.5*dr, 16.0]))
r_bin[0] = 0.0  # handle points less then the inner boundary

# our 2d density solution - flatten it for the calculation
Density  = np.zeros(X.shape)
Density  = Density.flatten()

# bin each point distance with the our solution
whichbin = np.digitize(D, r_bin)

# now grab corresponding density
for i,j in enumerate(whichbin):
    Density[i] = rho[j-1]

# reshape density
Density = Density.reshape(res, res)

plt.pcolor(X, Y, Density)
plt.xlabel("x", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.title("Density map for n=1")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
cbar = plt.colorbar()
cbar.set_label(r"$\theta = \rho/\rho_c$", fontsize=20)

plt.tight_layout()
plt.savefig("Star.png", format="png")
plt.show()
