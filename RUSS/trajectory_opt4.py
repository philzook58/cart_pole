import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt

g = 9.8
l = 1.0
dt = 0.1
lookahead = 100
#x[0] == x0
#x[1:] - x[:-2] ==



#nvars = 3 #? theta, thetadot, a
#x = np.zeros( (2,2*lookahead - 1))
#u = np.zeros( (1,2*lookahead - 1))
def f(x, u):
    #print(x)
    b = np.zeros_like(x)
    theta = x[0]
    dtheta = x[1]
    a = u[0]
    b[0] = dtheta
    b[1] = (a * np.cos(theta) - g * np.sin(theta)) / l
    return b
'''
def df(x, u):
    A = np.zeroes((lookahead,x.shape[1], x.shape[1]))
    B = np.zeroes((lookahead,x.shape[1], u.shape[1]))
    theta = x[:,0]
    dtheta = x[:,1]
    a = u[:,0]
    # dthetadot / dtheta
    A[:,0,1] = 1 
    # dtheta derviatvie.
    A[:,1,0] = (- a * np.sin(theta) - g * np.cos(theta)) / l
    B[:,1,0] = np.cos(theta) / l
    #b[1,:] = (a * np.cos(theta) - g * np.sin(theta)) / l
    return A, B
'''
def df(x, u):
    A = np.zeros((x.shape[0], x.shape[0]))
    B = np.zeros((x.shape[0], u.shape[0]))
    theta = x[0]
    dtheta = x[1]
    a = u[0]
    # dthetadot / dtheta
    A[0,1] = 1 
    # dtheta derviatvie.
    A[1,0] = (- a * np.sin(theta) - g * np.cos(theta)) / l
    B[1,0] = np.cos(theta) / l
    #b[1,:] = (a * np.cos(theta) - g * np.sin(theta)) / l
    return A, B



def linf(x, u, x2, u2, A, B, b):
    #b = f(x,u)
    #A, B = df(x,u)
    return b + A @ (x2 - x) + B * (u2 - u)
'''
def linf(x):
    b = f(x)
    A = df(x)
    return A, b 
    ''' 
#Constraint
#A * x = b
'''
def alpha(a,theta):
    return (a * np.cos(theta) - g * np.sin(theta)) / l

#derviatvies of dynamics
def dalpha(a,theta):
    dada = np.cos(theta) / l
    dadtheta = (- a * np.sin(theta) - g * np.cos(theta)) / l
    return dada, dadtheta

def linearApproxAlpha(a, theta): #takes Actual numbers
    dada, dadtheta  = dalpha(a,theta)
    def lin(dela, deltheta): #takes cvxpy variables
        return alpha(a, theta) + dada * dela + dadtheta * deltheta
    return lin
    '''

#np_a = np.zeros(lookahead)
#np_theta = np.zeros(lookahead)
#np_dtheta = np.zeros(lookahead)

#np_x[:,0] = np.linspace(0,np.pi,lookahead+1)
#np_x[:,1] = np.ones(lookahead+1) * np.pi / lookahead / dt
#lt.ion()
#plt.figure()




controls = []
constraints = []
thetas = []
dthetas = []
xs = []
As =[]
Bs = []
bs = []
cost = 0
'''
np_A = cvx.Parameter(2*lookahead+1, 2,2)
np_B = cvx.Parameter(2*lookahead+1, 2,1)
np_b = cvx.Parameter(2*lookahead+1, 2)
np_x = cvx.Parameter(2*lookahead+1, 2)
np_u = cvx.Parameter(lookahead+1,1)
'''
#print(np_u.shape)
#print(np_u.size)
#print(dir(np_u.size))
#np_x.value = np.zeros((2*lookahead+1, 2))
#np_u.value = np.zeros((lookahead+1, 1))
#np_A.value = np.zeros( (2*lookahead+1, 2,2) )
#np_B.value = np.zeros( (2*lookahead+1, 2,1) )
#np_b.value = np.zeros( (2*lookahead+1, 2))

#initiial condition constraints

A = cvx.Parameter(2,2)
As.append(A)
B = cvx.Parameter(2,1)
Bs.append(B)
b = cvx.Parameter(2)
bs.append(b)

x0 = cvx.Parameter(2)
u0 = cvx.Parameter(1)
x0s = []
u0s = []
u0s.append(u0)
x0s.append(x0)

x = cvx.Variable(2)
constraints.append(x[0] == 0)
constraints.append(x[1] == 0)

#dtheta = cvx.Variable()
#constraints.append(dtheta == 0)
xs.append(x)
#thetas.append(theta)
#dthetas.append(dtheta)
u = cvx.Variable(1)
constraints.append(u <= 13.0)   
constraints.append(u >= -13.0)
controls.append(u)
np_x = np.zeros((2*lookahead+1, 2))
np_u = np.zeros((lookahead+1, 1))

for i in range(lookahead):
    next_u = cvx.Variable(1)
    controls.append(next_u)

    # next time step variables
    next_x = cvx.Variable(2)
    half_x = cvx.Variable(2)

    half_A = cvx.Parameter(2,2)
    As.append(half_A)
    half_B = cvx.Parameter(2,1)
    Bs.append(half_B)
    half_b = cvx.Parameter(2)
    bs.append(half_b)

    half_x0 = cvx.Parameter(2)
    x0s.append(half_x0)

    next_x0 = cvx.Parameter(2)
    next_u0 = cvx.Parameter(1)
    u0s.append(next_u0)
    x0s.append(next_x0)

    next_A = cvx.Parameter(2,2)
    As.append(next_A)
    next_B = cvx.Parameter(2,1)
    Bs.append(next_B)
    next_b = cvx.Parameter(2)
    bs.append(next_b)

    #delthetan = cvx.Variable()
    #deldthetan = cvx.Variable()    
    #delthetas.append(delthetan)
    #deldthetas.append(deldthetan)
    xs.append(half_x)
    xs.append(next_x)
    #lin = linearApproxAlpha(a[i], theta[i])

    #Dynamics
    #b = f(np_x[i,:], np_u[i,:])
    #A, B = df(np_x[i,:], np_u[i,:])
    #bh = f(np_x[i+1,:], np_u[i+1,:])
    #Ah, Bh = df(np_x[i,:], np_u[i,:])
    #bn = f(np_x[i+2,:], np_u[i+2,:])
    #An, Bn = df(np_x[i+1,:], np_u[i,:])
    print(i)
    constraints.append(half_x == next_x/2 + x/2 + dt/8 * (linf(x0, u0,x, u, A, B, b ) - linf(next_x0, next_u0,next_x, next_u , next_A, next_B, next_b)))
    #print(B)
    #print(b)
    #print(b)
    #print(A)
    #print(B.reshape(-1))
    #print(np_u[i,:])
    #print(B@np_u[i,:])
    #print(np_x[i,:].shape)
    #print(A@np_x[i,:])
    #print(A@(x-np_x[i,:]))
    constraints.append(next_x - x ==  (linf(x0, u0, x, u , A, B, b ) + 4 * linf(half_x0, (u0 + next_u0) / 2, half_x, (u + next_u)/2 , half_A, half_B, half_b) + linf(next_x0, next_u0, next_x, next_u, next_A, next_B, next_b )  ) * dt / 6)
    #constraints.append(deldthetan == deldtheta + lin(at, deltheta) * dt)

    #conditions on allowable control
    constraints.append(next_u <= 8.0)   
    constraints.append(next_u >= -8.0)
    #trust regions
    #constraints.append(delthetan <= 0.5)   
    #constraints.append(delthetan >= -0.5)
    #constraints.append(deldthetan <= 0.5)   
    #constraints.append(deldthetan >= -0.5)


    #Cost calculation  
    cost = cost + cvx.huber( x[0] - np.pi, M=0.5) + 0.01 * cvx.huber(u)#+ (np.cos(np_x[2*i,:]) + 1) * (x[0] - np_x[2*i,:])  #+ cvx.square( x[0] - np.pi ) #+ cvx.square(u) #+ 0.1 * cvx.square(ut)
    # + cvx.square(np.cos(np_x[2*i,:])*(x - np_x[2*i,:]))  
    x = next_x
    u = next_u
    A = next_A
    B = next_B
    b = next_b

    x0 = next_x0
    u0 = next_u0



cost = cost + 100 * cvx.square( x[0] - np.pi )  # cvx.huber( x[0] - np.pi, M=0.4)
objective = cvx.Minimize(cost)
#print(objective)
#print(constraints)
prob = cvx.Problem(objective, constraints)

#print(sol)
#update by the del
#theta += np.array(list(map( lambda x: x.value, delthetas))) 
#print(x.value)
#print(constraints[0])
#np_x = np.array(list(map( lambda x: x.value, xs)))
#print(np_x.shape)
#np_x = np_x.reshape((-1,2))
#print(np_x.shape)
#np_u = np.array(list(map( lambda x: x.value, controls))).reshape((-1,1))
'''
plt.plot(np_x[::2,0])
plt.plot(np_x[::2,1])
plt.plot(np_u[:,0])

plt.show()
'''
#dtheta += np.array(list(map( lambda x: x.value, deldthetas))) 
#a += np.array(list(map( lambda x: x.value, controls)))

#print(np_u)
#np_A = np.zeros( (2*lookahead+1, 2,2) )
#np_B = np.zeros( (2*lookahead+1, 2,1) )
#np_b = np.zeros( (2*lookahead+1, 2))

print(len(bs))
print(len(As))
print(len(x0s))
print(len(u0s))
print(len(Bs))
for j in range(15):
    #np_x.value = np.zeros((2*lookahead+1, 2))
    #np_x[:,0] = np.linspace(0,np.pi,lookahead+1)
    #np_x[:,1] = np.ones(lookahead+1) * np.pi / lookahead / dt
    #lt.ion()
    #plt.figure()
    #np_u = np.zeros((lookahead+1, 1))
    #np_A = np.zeros( (2*lookahead+1, 2,2) )
    #np_B = np.zeros( (2*lookahead+1, 2,1) )
    #np_b = np.zeros( (2*lookahead+1, 2))
    #np_b.value = f(np_x.value, np_u.value)
    u0s[-1].value = np_u[-1]
    for i in range(lookahead):
        u0s[i].value = np_u[i]
        x0s[2*i].value = np_x[2*i]
        x0s[2*i+1].value = np_x[2*i+1]
        bs[2*i].value  = f(np_x[2*i,:], np_u[i,:])
        bs[2*i+1].value  = f(np_x[2*i+1,:], (np_u[i,:] + np_u[i+1,:]) / 2)
        As[2*i].value, Bs[2*i].value = df(np_x[2*i,:], np_u[i,:])
        As[2*i+1].value, Bs[2*i+1].value = df(np_x[2*i+1,:], (np_u[i,:] + np_u[i+1,:]) / 2)

    x0s[-1].value = np_x[-1]
    bs[-1].value  = f(np_x[-1,:], np_u[-1,:])
    #bs[2*i+1].value  = f(np_x[2*i+1,:], (np_u[i,:] + np_u[i+1,:]) / 2)
    As[-1].value, Bs[-1].value = df(np_x[-1,:], np_u[-1,:])
    #As[2*i+1].value, Bs[2*i+1].value = df(np_x[2*i+1,:], (np_u[i,:] + np_u[i+1,:]) / 2)

    sol = prob.solve(verbose=False)
    np_x = np.array(list(map( lambda x: x.value, xs))).reshape((-1,2))
    #print(np_x.shape)
    #print(np_x.shape)
    np_u = np.array(list(map( lambda x: x.value, controls))).reshape((-1,1))
    #np_u = np.zeros((lookahead+1, 1))





plt.plot(np_x[::2,0])
plt.plot(np_x[::2,1])
plt.plot(np_u[:,0])

plt.show()


#TODO
'''
We need to add cart position contraints
Parameters to hopefully speed up
Better initial guess.
Derivative of Cost
Is this actually working?
Maybe
Add no velocity constraint the the end

'''
