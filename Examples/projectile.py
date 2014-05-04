import numpy as np
import matplotlib.pyplot as plt

# parameters
g = 9.8     # graviational constants
v0 = 10.    # intial velocity
dt = 0.01   # time step 

# launch angle
angles = np.arange(30, 65, 5)

# solve for different angles
for theta in angles:

    # launch from the origin
    x = [0.0]
    y = [0.0]

    # intial velocity for give theta
    vx = [v0*np.cos(theta*np.pi/180)]
    vy = [v0*np.sin(theta*np.pi/180)]

    # integrate projectile motion equations
    # using euler method
    i = 0
    while y[i] >= 0.0:

        x.append(x[i] + vx[i]*dt)
        y.append(y[i] + vy[i]*dt)

        vx.append(vx[i])
        vy.append(vy[i] - g*dt)

        i += 1

    # plot solution for given theta
    plt.plot(x, y, label=`theta`+r"$^\circ$")

plt.ylim(0)             # set the lower y-axis to 0
l = plt.legend()        # show the legend
l.draw_frame(False)     # remove the box frame in legend
plt.show()
