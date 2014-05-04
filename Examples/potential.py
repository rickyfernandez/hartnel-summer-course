import numpy as np
import matplotlib.pyplot as plt

# create charee with value q and position x, y
np.random.seed(0)
q = np.random.uniform(-1, 1, 10)
x = np.random.uniform(0, 1, 10)
y = np.random.uniform(0, 1, 10)

# create grid
X = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(X,X); Y = np.flipud(Y)

# go through all cells and calculate the potential
# for each charge
V = np.zeros(X.shape)
for i in range(q.size):
    V += q[i]/np.sqrt((X-x[i])**2 + (Y-y[i])**2)

# plot the potential
plt.imshow(V, cmap=plt.get_cmap("gray"), origin='lower', extent=(0,1,0,1))
c = plt.colorbar() # add color bar
c.set_label(r"$4\pi\epsilon_0 V$", fontsize=16)

# plot equipotential lines
levels = np.linspace(.1*np.min(V), .1*np.max(V), 25)
plt.contour(V, levels=levels, origin='lower', extent=(0,1,0,1))

plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title(r"Electric Potential: $V=\sum_{i=1}^n\,\frac{1}{4\pi\epsilon_0}\,"
        r"\frac{q_i}{r}$", y=1.05, fontsize=16)
plt.subplots_adjust(top=0.8)
plt.show()
