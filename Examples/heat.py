import matplotlib.pyplot as plt
import numpy as np

N = 100  # number in dim
k = 5.0

x = np.linspace(-1,1,N)
dx = np.mean(np.diff(x))
X,Y = np.meshgrid(x,x)

dt = 0.25*dx**2/k

# set initial condition
#u = np.exp(-(X**2+Y**2)/0.25**2)
u = np.zeros(X.shape)
r = np.sqrt(X**2 + Y**2)
i = (.35 < r) & (r < .5)
u[i] = 1.0

#while t < tmax:
for i in range(100):
    
    plt.clf()
    plt.imshow(u, cmap=plt.get_cmap("hot"))
    plt.colorbar()
    plt.clim(0, 1)
    plt.savefig("density_" + `i`.zfill(4) + ".png")

    # update the solution
    u[1:-1, 1:-1] += dt*k*(u[:-2,1:-1] + u[2:,1:-1] +\
            u[1:-1,:-2] + u[1:-1,2:]-4*u[1:-1,1:-1])/dx**2

