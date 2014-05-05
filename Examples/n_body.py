import numpy as np
import matplotlib.pyplot as plt
import sys

class particle(object):

    def __init__(self, x, y, vx, vy, m, Id):
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.m = m
        self.Id = Id

    def move(self, f, dt):
        self.v += f*dt/self.m
        self.r += self.v*dt

    def force_from(self, b):
        G = 6.667E-11
        dr = self.r - b.r
        r = np.sqrt(np.dot(dr, dr))

        return -G*self.m*b.m*dr/r**3

class simulation(object):

    def __init__(self, fname):
        
        self.particle_list = []

        i = 0
        f = open(fname)
        for line in f:
            x, y, vx, vy, m = line.split()
            self.particle_list.append(
                    particle(float(x),float(y),float(vx),float(vy),float(m), i))
            i += 1

        self.N = len(self.particle_list)

    def run(self, num_it, dt):

        for i in xrange(num_it):
            self.integrate(dt)
            self.plot(i)

    def integrate(self, dt):

        force = np.zeros((self.N, 2))

        for p1 in self.particle_list:
            for p2 in self.particle_list:
                if p1.Id != p2.Id:
                    force[p1.Id,:] += p1.force_from(p2)

        for p in self.particle_list:
            p.move(force[p.Id,:], dt)


    def plot(self, i):

        for p in self.particle_list:
            plt.plot(p.r[0], p.r[1], 'ok')

        plt.xlim(-1.25E11, 1.25E11)
        plt.ylim(-1.25E11, 1.25E11)
        plt.savefig("n_body_" + `i`.zfill(4) + ".png")
        plt.clf()
            

if __name__ == "__main__":

    filename = "body.txt"
    number_of_iter = 800
    dt = 20000 

    sim = simulation(filename)
    sim.run(number_of_iter, dt)
