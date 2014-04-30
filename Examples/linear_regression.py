import numpy as np
import matplotlib.pyplot as plt

def cost_function(X, y, theta):

    m = y.shape[0]
    error = np.dot(X, theta) - y

    J = 0.5*np.dot(error, error)/m
    return J

def gradient_descent(X, y, theta, alpha, num_iterations):

    m = y.shape[0]
    X_T = X.T

    for i in xrange(num_iterations):
        error = np.dot(X, theta) - y
        theta -= alpha*np.dot(X_T, error)/m

# generate data - y = f(x) + e: f(x) = 2x + 1
np.random.seed(1)
x = np.random.normal(1., .5, 100)
y = 2.*x + 1. + np.random.normal(0., .5, 100)

alpha = 0.10          # strength of descent
theta = np.zeros(2)   # initial theta

# find theta that minimizes the cost function
X = np.array([np.ones(x.size), x]).T
gradient_descent(X, y, theta, alpha, 3000)

# create grid of cost function for different theta
theta0 = np.linspace(-4, 5, 300)
theta1 = np.linspace(-2, 8, 300)
J_grid = np.zeros((300,300))

for i, t0 in enumerate(theta0):
    for j, t1 in enumerate(theta1):
        J_grid[i,j] = cost_function(X, y, np.array([t0, t1]))

# plot linear regression with the data
plt.subplot(1,2,1)
plt.plot(x, y, 'or')
plt.plot(x, 2.*x + 1, 'b')
plt.plot(x, np.dot(X, theta), 'r')

l = plt.legend(["Data points", r"$f(x)$", "Regression"], prop={'size':12})
l.draw_frame(False)

plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.title(r"Linear Regression: $y=f(x)+\epsilon$")

plt.xlim(0, 2)
plt.ylim(0, 7)

# plot the space of possible theta's with the best choice
plt.subplot(1,2,2)
plt.contour(theta0, theta1, J_grid.T, np.logspace(-2, 3, 10))
plt.plot(theta[0], theta[1], 'or', markersize=5)

plt.xlabel(r"$\theta_0$", fontsize=15)
plt.ylabel(r"$\theta_1$", fontsize=15)
plt.title(r"Cost Function $J(\theta)$")

plt.tight_layout()
plt.show()
