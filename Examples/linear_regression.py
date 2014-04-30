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

alpha = 0.01          # strength of descent
theta = np.zeros(2)   # initial theta

# find theta that minimizes the cost function
X = np.array([np.ones(x.size), x]).T
gradient_descent(X, y, theta, alpha, 1500)

plt.plot(x, y, 'or')
plt.plot(x, 2.*x + 1, 'b')
plt.plot(x, np.dot(X, theta), 'r')

l = plt.legend(["Data points", r"$f(x)$", "Regression"], prop={'size':12})
l.draw_frame(False)

plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Linear Regression: $y=f(x)+\epsilon$")

plt.xlim(0, 2)
plt.ylim(0, 7)
plt.show()
