import sparsegrad.forward as forward
import numpy as np
import osqp
import scipy.sparse as sparse


class MPC():
	def __init__(self, x0, v0, theta0, thetadot0):
		self.N = 1000
		self.NVars  = 5
		T = 5.0
		dt = T/self.N
		self.dtinv = 1./dt
		#Px = sparse.eye(N)
		#sparse.csc_matrix((N, N)) 
		# The three deifferent weigthing matrices for x, v, and external force
		reg = sparse.eye(self.N)*0.01
		# sparse.diags(np.arange(N)/N)
		P = sparse.block_diag([reg,reg ,sparse.eye(self.N), 1*reg,1*reg])
		#P[N,N]=10
		THETA = 2
		q = np.zeros((self.NVars, self.N))
		q[THETA,:] = np.pi
		#q[N,0] = -2 * 0.5 * 10
		q = q.flatten()
		q= -P@q
		#u = np.arr

		x = np.random.randn(self.N,self.NVars).flatten()
		#x = np.zeros((N,NVars)).flatten()
		#v = np.zeros(N)
		#f = np.zeros(N)


		#print(f(ad.seed(x)).dvalue)




		A, l, u = self.getAlu(x, x0, v0, theta0, thetadot0)
		self.m = osqp.OSQP()
		self.m.setup(P=P, q=q, A=A, l=l, u=u) #  **settings
		self.results = self.m.solve()
		print(self.results.x)
		for i in range(100):
			self.update(x0, v0, theta0, thetadot0)


	def update(self, x0, v0,theta0, thetadot0):
		A, l, u = self.getAlu(self.results.x, x0, v0, theta0, thetadot0)
		print(A.shape)
		#print(len(A))
		self.m.update(Ax=A.data, l=l, u=u)
		self.results = self.m.solve()
		return self.results.x[4*self.N+1]



	def constraint(self, var, x0, v0, th0, thd0):
		#x[0] -= 1
		#print(x[0])
		g = 9.8
		L = 0.5
		gL = g * L
		m = 1.0 # doesn't matter
		I = L**2 / 3
		Iinv = 1.0/I
		dtinv = self.dtinv
		N = self.N

		x = var[:N]
		v = var[N:2*N]
		theta = var[2*N:3*N]
		thetadot = var[3*N:4*N]
		a = var[4*N:5*N]
		dynvars = (x,v,theta,thetadot)
		xavg, vavg, thetavg, thdotavg = map(lambda z: (z[0:-1]+z[1:])/2, dynvars)
		dx, dv, dthet, dthdot = map(lambda z: (z[1:]-z[0:-1])*dtinv, dynvars)
		vres = dv - a[1:]
		xres = dx - vavg
		torque = -gL*np.sin(thetavg) + a[1:]*L*np.cos(thetavg)
		thetdotres = dthdot - torque*Iinv
		thetres = dthet - thdotavg

		return x[0:1]-x0, v[0:1]-v0, theta[0:1]-th0, thetadot[0:1]-thd0, xres,vres, thetdotres, thetres
		#return x[0:5] - 0.5



	def getAlu(self, x, x0, v0, th0, thd0):
		N = self.N
		gt = np.zeros((2,N))
		gt[0,:] = 0.1 # x is greaer than 0
		gt[1,:] = -1 #veclotu is gt -1m/s
		gt = gt.flatten()
		lt = np.zeros((2,N))
		lt[0,:] = 0.8
		lt[1,:] = 1 # velocity less than 1m/s
		lt = lt.flatten()

		z = sparse.bsr_matrix((N, N))
		ineqA = sparse.bmat([[sparse.eye(N),z,z,z,z],[z,sparse.eye(N),z,z,z]]) #.tocsc()
		#print(ineqA.shape)
		#print(ineqA)
		cons = self.constraint(forward.seed_sparse_gradient(x), x0, v0, th0, thd0)
		A = sparse.vstack(map(lambda z: z.dvalue, cons)) #  y.dvalue.tocsc()
		#print(A.shape)
		totval = np.concatenate(tuple(map(lambda z: z.value, cons)))
		temp = A@x - totval


		A = sparse.vstack((A,ineqA)).tocsc()

		#print(tuple(map(lambda z: z.value, cons)))
		#print(temp.shape)
		#print(lt.shape)
		#print(gt.shape)
		u = np.concatenate((temp, lt))
		l = np.concatenate((temp, gt))
		return A, l, u





