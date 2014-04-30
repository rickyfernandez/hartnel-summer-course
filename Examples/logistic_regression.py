import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    
    g = 1./(1. + np.exp(-z))
    return g

def cost_function(theta, X, y):

    # number of training examples
    m = y.shape[0]
    z = np.dot(X,theta)
    h = sigmoid(z)

    J = np.sum(-y*np.log(h) - (1.-y)*np.log(1.-h))/m

    return J

def gradient(theta, X, y):

    # number of training examples
    m = y.shape[0]
    h = sigmoid(np.dot(X,theta))

    grad = np.dot(X.T, h-y)/m

    return grad


# load test scores to numpy arrays
data = np.loadtxt("ex2data1.txt", delimiter=",")

x = data[:,0:2]
y = data[:,2]

# classify the points
i = y == 1 # admitted
j = ~i     # not admitted

initial_theta = np.zeros(3)   # initial theta

## find theta that minimizes the cost function
m = x.shape[0] # number of training examples
X = np.append(np.ones(m)[:,np.newaxis], x, 1)

theta = op.minimize(fun=cost_function, x0=initial_theta,
        args=(X,y), method="TNC", jac=gradient)

# find decision boundary by setting z=x*theta=0
# that is the values of x such that p=0.5
x_b = np.array([np.min(x[:,0])-2, np.max(x[:,0])+2])
y_b = -(theta.x[0]+theta.x[1]*x_b)/theta.x[2]

# find accuracy of logistic regression
p = sigmoid(np.dot(X, theta.x)) >= 0.5
print "\naccuracy of logistic regression: %.2f%%" % (np.sum(p == y)/float(m))

plt.plot(x_b, y_b)
plt.plot(x[i,0], x[i,1], 'oy')
plt.plot(x[j,0], x[j,1], 'ok')
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Logistic Regression")
plt.xlim(np.min(x[:,0]), np.max(x[:,0]))
plt.ylim(np.min(x[:,1]), np.max(x[:,1]))
plt.show()
