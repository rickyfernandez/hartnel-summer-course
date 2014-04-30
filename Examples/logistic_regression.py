import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    
    g = 1./(1. + np.exp(-z))
    return g

def cost_function(theta, X, y):

    m = y.shape[0]
    z = np.dot(X,theta)
    h = sigmoid(z)

    J = np.sum(-y*np.log(h) - (1.-y)*np.log(1.-h))/m
    return J

def gradient(theta, X, y):

    m = y.shape[0]
    h = sigmoid(np.dot(X,theta))

    grad = np.dot(X.T, h-y)/m
    return grad


# load test scores to numpy arrays
data = np.loadtxt("ex2data1.txt", delimiter=",")

x = data[:,0:2]  # features
y = data[:,2]    # outputs

# classify the points
i = y == 1 # admitted
j = ~i     # not admitted

initial_theta = np.zeros(3)

# number of features
m = x.shape[0]

# add intercept to features
X = np.append(np.ones(m)[:,np.newaxis], x, 1)

# find theta that minimizes the cost function
theta = op.minimize(fun=cost_function, x0=initial_theta,
        args=(X,y), method="TNC", jac=gradient)

# find decision boundary by setting z=x*theta=0 that is 
# the values of x such that the probability p=sigmoid(z)=0.5
x_b = np.array([np.min(x[:,0])-2, np.max(x[:,0])+2])
y_b = -(theta.x[0]+theta.x[1]*x_b)/theta.x[2]

# find accuracy of logistic regression
p = sigmoid(np.dot(X, theta.x)) >= 0.5
print "\naccuracy of logistic regression: %.1f%%" % (np.sum(p == y)/float(m)*100.)

plt.plot(x[i,0], x[i,1], 'ob', alpha=0.7)
plt.plot(x[j,0], x[j,1], 'or', alpha=0.7)
plt.plot(x_b, y_b, "k")

plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Logistic Regression")
plt.legend(["Admitted", "Not Admitted", "Regression"], prop={'size':12})

plt.xlim(30, 100)
plt.ylim(30, 100)
plt.show()
