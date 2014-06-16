import numpy as np
import matplotlib.pyplot as plt

# parameters
g = 9.8
l = 1.0
N = 50
tmax = 3.

dt = tmax/N

# initial conditions
theta0 = 25 * np.pi / 180.
omega0 = 0.
t0     = 0.

# store solution
theta = np.zeros(N)
omega = np.zeros(N)
time  = np.zeros(N)


theta[0] = theta0
omega[0] = omega0
time[0]  = t0

i=0
while i < N-1:

    # better method than the euler
    omega[i+1] = omega[i] - g/l*theta[i]*dt
    theta[i+1] = theta[i] + omega[i+1]*dt
    time[i+1] = time[i] + dt

    # map to rectangular coords
    x = l*np.sin(theta[i+1])
    y = -l*np.cos(theta[i+1])

    # create figure to plot 
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot([0, x], [0, y], 'k', alpha=0.6, lw=3)
    plt.scatter(x, y, c='r', s=1000)
    plt.axhline(-1, c='k', lw=5)

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("Pendulum")

    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.2,0)

    plt.subplot(1,2,2)
    plt.scatter(time[i+1], theta[i+1], c='r', s=80)
    plt.plot(time[:i+2], theta[:i+2])

    plt.xlabel("time")
    plt.ylabel(r"$\theta$", fontsize=18)
    plt.title("Angle")
    plt.xlim(t0, tmax)
    plt.ylim(-1.2*theta0, 1.2*theta0)

    plt.tight_layout()
    plt.savefig("pen_" + str(i).zfill(4))
    plt.clf()

    i += 1
