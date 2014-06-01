import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.integrate import ode

def F(t, wt2, nu, K, K2, n, KX, KY):

    # reshape 1d array into 2d array
    #print "type: ", type(wt2)
    wt = wt2.reshape(n, n)

    # calculate psi in fourier space
    psit = -wt/K

    # calculate the derivatives of psi and w
    # in real space
    psi_x = np.real(fft.ifft2(1.j*KX*psit))
    psi_y = np.real(fft.ifft2(1.j*KY*psit))
    w_x   = np.real(fft.ifft2(1.j*KX*wt))
    w_y   = np.real(fft.ifft2(1.j*KY*wt))

    return fft.fft2(w_x*psi_y - w_y*psi_x).flatten() - nu*K2*wt2

# time slices
t_0 = 0.; t_final = 80
#tspan = np.linspace(t_0, t_final)
#dt = np.mean(np.diff(tspan))
dt = 1.0
nu = 0.001

L = 20
n = 128

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
K2 = K.flatten()

# initial condition
w = np.exp(-0.25*X**2 - 2.*Y**2)
wt = fft.fft2(w)
wt2 = wt.flatten()

# setup the integrator
#sol = ode(F).set_integrator('zvode', atol=1.E-10, rtol=1.E-10)
sol = ode(F).set_integrator('dopri5', atol=1.E-10, rtol=1.E-10)
sol.set_initial_value(wt2, t_0)
sol.set_f_params(nu, K, K2, n, KX, KY)


index = 0
# solve using runge kutta
while sol.successful() and sol.t < t_final:

    print "time:", sol.t

    sol.integrate(sol.t + dt)
    w = np.real(fft.ifft2(sol.y.reshape(n,n)))

    plt.pcolor(X, Y, w)
    plt.xlim(-.5*L, .5*L)
    plt.ylim(-.5*L, .5*L)
    plt.colorbar()
    plt.clim(0., 1.)

    plt.savefig("Vorticity_" + `index`.zfill(4))
    plt.clf()
    index += 1
