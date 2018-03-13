import numpy as np
import cvxpy as cvx

#Hamilton Jacobi

# dJ/dt = g + dJ/dx * dx/dt
thetas = np.linspace(-np.pi, np.pi , 100).reshape(1,-1)
thetadot = np.linspace(-10, 10 , 100).reshape(-1,1)
#xs = np.linspace(-1,1,100)
#xdots = np.linspace()

def cost(theta):
	return (theta - 0)**2

def alpha(theta, tau):
	return np.sin(theta) + tau

#tss, xss, xdss, np.meshgrid()

gradJ/dx dot F > cost

J > 0
Min J?
