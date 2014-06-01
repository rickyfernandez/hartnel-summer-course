import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def F(wt, t, nu, KX, KY, K):

    # calculate psi in fourier space
    psit = -wt/K

    # calculate the derivatives of psi and w
    # in real space
    psi_x = np.real(fft.ifft2(1.j*KX*psit))
    psi_y = np.real(fft.ifft2(1.j*KY*psit))
    w_x   = np.real(fft.ifft2(1.j*KX*wt))
    w_y   = np.real(fft.ifft2(1.j*KY*wt))

    return fft.fft2(w_x*psi_y - w_y*psi_x) - nu*K*wt

# time slices
tn = 40
tspan = np.linspace(0,10,tn)
dt = np.mean(np.diff(tspan))
nu = 0.001

L = 20
n = 246

# remember fft ignores the last point because it
# assumes the function is periodic: f[0] = f[-1]
x = np.linspace(-L/2, L/2, n, endpoint=False)
y = x.copy()

# create the wave numbers
kx = (2*np.pi/L)*np.append(np.arange(0,n/2), np.arange(-n/2,0))
kx[0] = 1.0E-6 # to avoid division by zero
ky = kx.copy()

# now create 2 grid
X, Y = np.meshgrid(x,y)
KX, KY = np.meshgrid(kx,ky)

K = KX**2 + KY**2

# initial condition
w = np.exp(-2.*X**2 - Y**2/20.)
wt = fft.fft2(w)

# solve using runge kutta
n = 0
t = 0
for i in tspan:

        # second order runge kutta method
        k1 = dt*F(wt, t, nu, KX, KY, K)
        k2 = dt*F(wt + 0.5*k1, t + 0.5*dt, nu, KX, KY, K)
        k3 = dt*F(wt + 0.5*k2, t + 0.5*dt, nu, KX, KY, K)
        k4 = dt*F(wt + k3, t + dt, nu, KX, KY, K)

        # update the solution
        wt += (k1 + 2.*k2 + 2.*k3 + k4)/6.0
        t += dt

        w = np.real(fft.ifft2(wt))

        plt.pcolor(X, Y, w)
        plt.xlim(-.5*L, .5*L)
        plt.ylim(-.5*L, .5*L)
        plt.colorbar()
        plt.clim(0, 1.)

        plt.savefig("Vorticity_" + `n`.zfill(4))
        plt.clf()
        n += 1
