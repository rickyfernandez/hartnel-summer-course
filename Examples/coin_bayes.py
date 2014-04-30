import matplotlib.pyplot as plt
from random import choice
import numpy as np
import scipy.integrate

def LogLikelihood(N, R, H):
    # N number of flips, R number of heads, and H belief of bias in heads
    p = np.zeros(H.size)

    # find non zero values
    i = (H > 0) & ((1-H) > 0)

    # take natural log of likihood ignoring zero values
    p[i]  = R*np.log(H[i]) + (N-R)*np.log(1.0-H[i])
    p[~i] = 0

    return p, i

# unfair coin - heads 0.25 of the time
coin = [0, 1, 1, 1]
data = np.array([choice(coin) for i in xrange(1, 4097)])

# our initial ignorance of the coin - equal
# probabilty for any bias
prior = np.ones(1000)
bias  = np.linspace(0, 1, len(prior))
posterior = []; posterior.append(prior)

events = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for flips in events:

    heads = sum(data[0:flips] == 1)
    post, i = LogLikelihood(flips, heads, bias)

    # subtract max value
    j = np.argmax(post[i])
    post[i] -= post[j]

    # take anti-log
    post[i] = np.exp(post[i])

    # normalize the probability
    norm  = scipy.integrate.simps(post, bias)
    post /= norm

    posterior.append(post)

# calculate mean and standard deviation 
mu = scipy.integrate.simps(posterior[-1]*bias, bias)
vr = scipy.integrate.simps(posterior[-1]*(mu-bias)**2, bias)
sd = np.sqrt(vr)

print "Bayes-The coin bias is", mu
print "Bayes-limits:", mu+2.0*sd, mu-2.0*sd
print "Frequency-The coin bias is", sum(data==1)/float(len(data))


# plot the evolution of the bias
fig, axes = plt.subplots(nrows=5, ncols=3)

for i, ax in zip(posterior, axes.flat):
    ax.plot(bias, i)
    ax.axvline(.75, color = "r", ls="--")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])

plt.show()
