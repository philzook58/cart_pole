import sparsegrad.forward as forward
import numpy as np
import osqp
import scipy.sparse as sparse
import matplotlib.pyplot as plt

N = 100
NVars = 5
# x, x_dot, theta, theta_dot, accel

P = sparse.block_diag([sparse.eye(N)*0.01, sparse.eye(N)*0.01, sparse.diags(np.arange(N)/N), sparse.eye(N)*0.01, sparse.eye(N)*0.1])

set_point_theta = np.pi

q = np.zeros((NVars, N))
q[2,:] = -set_point_theta*np.arange(N)/N

q = q.flatten()

x = np.zeros((N,NVars)).flatten()

def constraint(var, x0, x_dot0, theta0, theta_dot0):
	x = var[:N]
	x_dot = var[N:2*N]
	theta = var[2*N:3*N]
	theta_dot = var[3*N:4*N]
	a = var[4*N:5*N]

	avg_x = (x[0:-1]+x[1:])/2
	avg_x_dot = (x_dot[0:-1]+x_dot[1:])/2
	avg_theta = (theta[0:-1]+theta[1:])/2
	avg_theta_dot = (theta_dot[0:-1]+theta_dot[1:])/2

	dt1 = 1/0.1

	dx = (x[0:-1]-x[1:]) * dt1
	dx_dot = (x_dot[0:-1]-x_dot[1:]) * dt1
	dtheta = (theta[0:-1]-theta[1:]) * dt1
	dtheta_dot = (theta_dot[0:-1]-theta_dot[1:]) * dt1
	f = a[1:] #return v - f()
	t = -np.sin(avg_theta) + a[1:] * np.cos(avg_theta)

	x_res = dx - avg_x_dot
	x_dot_res = dx_dot - f
	theta_res = dtheta - avg_theta_dot
	theta_dot_res = dtheta_dot - t

	return x[0:1]-x0, x_dot[0:1]-x_dot0, theta[0:1]-theta0, theta_dot[0:1]-theta_dot0, x_res, x_dot_res, theta_res, theta_dot_res 


cons = constraint(forward.seed_sparse_gradient(x), 0, 0, 0, 0)
A = sparse.vstack(map(lambda z: z.dvalue, cons)).tocsc() 

totval = np.concatenate(tuple(map(lambda z: z.value, cons)))
temp = A@x - totval
u = temp
l = temp

m = osqp.OSQP()
m.setup(P=P, q=q, A=A, l=l, u=u)
results = m.solve()
print(results.x)

plt.plot(results.x[:N], label="x")
plt.plot(results.x[N:2*N], label="x_dot")
plt.plot(results.x[2*N:3*N], label="theta")
plt.plot(results.x[3*N:4*N], label="theta_dot")
plt.plot(results.x[4*N:5*N], label="f")
plt.legend()


plt.show()