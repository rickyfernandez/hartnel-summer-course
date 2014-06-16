import matplotlib.pyplot as plt
import numpy as np

L = 1.0     # length of the domain
n = 256     # discretize the domain to n points

# create the domain
x = np.linspace(0, L, n, endpoint=False)
y = np.sin(2*np.pi*4*x)      # signal

# generate the wave numbers
k = 2*np.pi/L * np.append(np.arange(0,n/2), np.arange(-n/2,0))
ut = np.fft.fft(y)           # fourier transform of y

fig, axes = plt.subplots(1,2)

# plot the signal
axes[0].plot(x,y,'k')        # plot original curve
axes[0].set_xlabel("x")
axes[0].set_title("signal")

# plot the power spectrum
ps = np.absolute(0.5*ut/n)   # normalized power spectr

# plot power spectrumum
axes[1].plot(k, ps, 'or', label="power spectrum")
axes[1].set_xlabel("k")
#axes[1].set_title("power spectrum")

# plot the expected frequencies
k1 = 2*np.pi*4; k2 = -k1
axes[1].vlines(k1, 0, np.max(np.absolute(ps)), 'b', '--',
        label="expected")
axes[1].vlines(k2, 0, np.max(np.absolute(ps)), 'b', '--')
axes[1].set_xlim(-50, 50)
axes[1].set_ylim(-0.01, 1.25*np.max(ps))
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig("fft_example_1")
plt.show()
