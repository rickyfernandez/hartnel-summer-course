import numpy as np
import matplotlib.pyplot as plt

# parameters
g = 9.8     # gravitational constant
l = 1.0     # length of pendulm

# period of pendulm T = 2pi(l/g)^2
# for our values T=2.01s
N = 500     # number of steps
tmax = 10.  # total time of simulation

dt = tmax/N

# initial theta in radians
theta0 = 0.2

theta = np.zeros(N)
omega = np.zeros(N)
t     = np.zeros(N)

omega[0] = theta0

# do the integration
for i in xrange(N-1):

    omega[i+1] = omega[i]-g/l*theta[i]*dt
    theta[i+1] = theta[i]+omega[i+1]*dt
    t[i+1] = t[i]+dt

# calculate the energy
E = 0.5*omega**2 + 0.5*g*theta**2

plt.subplot(1,2,1)
plt.plot(t, omega)
plt.xlabel(r"Time $(s)$", fontsize=12)
plt.ylabel(r"$\theta$ (rads)", fontsize=12)

plt.subplot(1,2,2)
plt.plot(t, E)
plt.xlabel(r"Time $(s)$", fontsize=12)
plt.ylabel(r"Energy $(J)$", fontsize=12)

plt.tight_layout()
plt.show()
