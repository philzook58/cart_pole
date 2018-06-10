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

def linf(x, u, x2, u2):
    b = f(x,u)
    A, B = df(x,u)
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
np_x = np.zeros((2*lookahead+1, 2))
#np_x[:,0] = np.linspace(0,np.pi,lookahead+1)
#np_x[:,1] = np.ones(lookahead+1) * np.pi / lookahead / dt
#lt.ion()
#plt.figure()
np_u = np.zeros((lookahead+1, 1))
for j in range(15):
    controls = []
    constraints = []
    thetas = []
    dthetas = []
    xs = []
    cost = 0
    

    #initiial condition constraints

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

    for i in range(lookahead):
        next_u = cvx.Variable(1)
        controls.append(next_u)

        # next time step variables
        next_x = cvx.Variable(2)
        half_x = cvx.Variable(2)

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
        constraints.append(half_x == next_x/2 + x/2 + dt/8 * (linf(np_x[2*i,:], np_u[i,:],x, u ) - linf(np_x[2*i+2,:], np_u[i+1,:],next_x, next_u )))
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
        constraints.append(next_x - x ==  (linf(np_x[2*i,:], np_u[i,:], x, u ) + 4 * linf(np_x[2*i+1,:], (np_u[i,:] + np_u[i+1,:]) / 2, half_x, (u + next_u)/2) + linf(np_x[2*i+2,:], np_u[i+1,:], next_x, next_u )  ) * dt / 6)
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



    cost = cost + 100 * cvx.square( x[0] - np.pi )  # cvx.huber( x[0] - np.pi, M=0.4)
    objective = cvx.Minimize(cost)
    #print(objective)
    #print(constraints)
    prob = cvx.Problem(objective, constraints)
    sol = prob.solve(verbose=True)
    #print(sol)
    #update by the del
    #theta += np.array(list(map( lambda x: x.value, delthetas))) 
    #print(x.value)
    #print(constraints[0])
    np_x = np.array(list(map( lambda x: x.value, xs)))
    print(np_x.shape)
    np_x = np_x.reshape((-1,2))
    print(np_x.shape)
    np_u = np.array(list(map( lambda x: x.value, controls))).reshape((-1,1))
    '''
    plt.plot(np_x[::2,0])
    plt.plot(np_x[::2,1])
    plt.plot(np_u[:,0])

    plt.show()
    '''
    #dtheta += np.array(list(map( lambda x: x.value, deldthetas))) 
    #a += np.array(list(map( lambda x: x.value, controls)))
 
#print(np_u)

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
'''
