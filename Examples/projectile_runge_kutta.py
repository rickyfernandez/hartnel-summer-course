import numpy as np
import matplotlib.pyplot as plt

def f(r, t):

    g = 9.8

    x  = r[0]
    y  = r[1]
    vx = r[2]
    vy = r[3]

    fx  = vx
    fy  = vy
    fvx = 0.
    fvy = -g

    return np.array([fx, fy, fvx, fvy])

# parameters
h = 0.1 
g = 9.8
t = 0.
x0 = 0.
y0 = 0.
v0 = 10.
theta = 45.*np.pi/180.

# solution arrays
x  = [x0]; y = [y0]
vx = [v0*np.cos(theta)]
vy = [v0*np.sin(theta)]

time = [t]

# initial vector
r = np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)])

while r[1] >= 0.:

    # runge kutta coefficients
    k1 = h*f(r, t)
    k2 = h*f(r+0.5*k1, t+0.5*h)
    k3 = h*f(r+0.5*k2, t+0.5*h)
    k4 = h*f(r+k3, t+h)

    # update the solution 
    r += (k1+2*k2+2*k3+k4)/6.0
    t += h

    # store the solution for position
    x.append(r[0])
    y.append(r[1])

    # store the solution for velocity
    vx.append(r[2])
    vy.append(r[3])

    # store the time
    time.append(t)

# plot the true solution
# x = x0 + v*t y = y0 + v*t - 0.5*g*t**2
time = np.array(time)
xsol = x0 + v0*np.cos(theta)*time
ysol = y0 + v0*np.sin(theta)*time - 0.5*g*time**2

plt.plot(x,y,'o')
plt.plot(xsol, ysol, 'r')
plt.show()
