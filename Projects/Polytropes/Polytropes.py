import matplotlib.pyplot as plt
import numpy as np
import sys

def f(r, t, n):
    x = r[0]
    y = r[1]
    
    fx = y
    fy = -(2.0/t)*y-x**n
    return np.array([fx, fy])

def RK4(r, t, h, n):

    xp = []; yp = []; tp = []
    xp.append(r[0]); yp.append(r[1]); tp.append(t)

    # integrate until the density vanishes
    while r[0] > 1.0E-10:

        # runge kutta coefficients
        k1 = h*f(r, t, n)
        k2 = h*f(r+0.5*k1, t+0.5*h, n)
        k3 = h*f(r+0.5*k2, t+0.5*h, n)
        k4 = h*f(r+k3, t+h, n)

        # update the solution 
        r += (k1+2*k2+2*k3+k4)/6.0
        t += h

        # store the solution
        xp.append(r[0])
        yp.append(r[1])
        tp.append(t)

    return np.array(xp), np.array(yp), np.array(tp)


if __name__ == "__main__":
   
    # grab problem number from the terminal prompt
    problem = int(sys.argv[1])

    if problem == 1:

        # parameters and initial conditions
        h = 0.1     # step size
        t0 = h      # initial radius

        # hold the solutions for eacn index n
        theta_sol = []
        xi_sol    = []

        # solve the problem for different index n's
        for n in [0, 1]:

            # boundary conditions
            r0 = np.array([1.0, 0.0])
            # 4th order runge kutta integration
            x, y, t = RK4(r0, t0, h, n)

            # store the solution
            theta_sol.append(x)
            xi_sol.append(t)
       

        # generate true solutions
        sol = []
        # for n = 0 case
        sol.append(1-xi_sol[0]**2/6.0)
        # for n = 1 case
        sol.append(np.sin(xi_sol[1])/xi_sol[1])

        # plot the exact and numerical solution for n = 0
        plt.plot(xi_sol[0], sol[0], 'r', linewidth=1.5, alpha=0.4 )
        plt.plot(xi_sol[0], theta_sol[0], 'ro', alpha=0.7)

        # plot the exact and numerical solution for n = 1
        plt.plot(xi_sol[1], sol[1], 'b', linewidth=1.5, alpha=0.4)
        plt.plot(xi_sol[1], theta_sol[1], 'bo', alpha=0.7)

        l = plt.legend(["n=0: Eact Solution", "n=0: Numerical Solution",
            "n=1: Eact Solution", "n=1: Numerical Solution"], prop={'size':12})
        l.draw_frame(False)

        plt.xlabel(r"$\xi$", fontsize=20)
        plt.ylabel(r"$\theta$", fontsize=20)
        plt.title("")
        
        plt.xlim(0, 4)
        plt.ylim(0, 1)

        plt.savefig("Polytropes.pdf", format="pdf")
        plt.show()

    #if problem == 2:

        #print tpoints[-1]**2.0*ypoints[-1]
