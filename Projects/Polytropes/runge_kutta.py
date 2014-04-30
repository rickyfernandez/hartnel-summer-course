import numpy as np

def f(r, t):
    x = r[0]
    y = r[1]
    fx = "function for the first variable"
    fy = "function for the second variable"
    return np.array([fx, fy])

# parameters
h = "step size"
t = "initial step"

# solution arrays
xp = []; yp = []; tp = []

r = np.array(["your initial conditions"])
while "radius" < "radius criteria":

    # runge kutta coefficients
    k1 = h*f(r, t)
    k2 = h*f(r+0.5*k1, t+0.5*h)
    k3 = h*f(r+0.5*k2, t+0.5*h)
    k4 = h*f(r+k3, t+h)

    # update the solution 
    r += (k1+2*k2+2*k3+k4)/6.0
    t += h

    # store the solution
    xp.append(r[0])
    yp.append(r[1])
    tp.append(t)

