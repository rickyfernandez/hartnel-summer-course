import matplotlib.pyplot as plt
import numpy as np

L = 5.0     # length of the domain
n = 256     # discretize the domain to n points

# create the domain
x = np.linspace(0, L, n, endpoint=False)
w1 = 5.0    # wavelength 1
w2 = 1.0    # wavelength 2

# signal with wavelenth 1 and 2
y = np.sin(2*np.pi*x/w1) + np.cos(2*np.pi*x/w2)

# generate the wave numbers
k = 2*np.pi/L * np.append(np.arange(0,n/2), np.arange(-n/2,0))
ut  = np.fft.fft(y)           # fourier transform of y

fig, axes = plt.subplots(3,1)

# plot the signal
axes[0].plot(x,y,'k', lw=1.5)
axes[0].set_xlabel("x")
axes[0].set_title("signal")

# plot the power spectrum
ps = np.absolute(ut[0:n/2]/n)  # normalize the power spectrum
axes[1].stem(k[0:n/2], ps)
axes[1].set_xlim(0,10)
axes[1].set_ylim(0, 1.1*np.max(ps))
axes[1].set_xlabel("k")
axes[1].set_title("power spectrum")

# filter the signal
ulo = ut.copy()
ulo[np.abs(k) > 3] = 0
uhi = ut.copy()
uhi[np.abs(k) < 3] = 0

# inverse fourier transform
ylo = np.real(np.fft.ifft(ulo))
yhi = np.real(np.fft.ifft(uhi))

# plot signal and decomposition
axes[2].plot(x,y,   'k', lw=1.5)
axes[2].plot(x,ylo, 'b', alpha=0.4)
axes[2].plot(x,yhi, 'r', alpha=0.4)
axes[2].set_xlabel("x")
axes[2].set_title("signal and decomposition")

plt.tight_layout()
plt.show()
