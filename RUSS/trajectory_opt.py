import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt

g = 9.8
l = 1.0
dt = 0.01



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
lookahead = 100
a = np.zeros(lookahead)
#theta = np.ones(lookahead+1) * (np.pi + 0.4 )
theta = np.zeros(lookahead+1) 
dtheta = np.zeros(lookahead+1)
palpha = cvx.Parameter(lookahead)
pdadtheta = cvx.Parameter(lookahead)
pdada = cvx.Parameter(lookahead)

#palpha.value = alpha(a, theta)
#dadtheta.value, pdada.value = dalpha(a, theta)



controls = []
constraints = []
delthetas = []
deldthetas = []

cost = 0


#initiial condition constraints

deltheta = cvx.Variable()
constraints.append(deltheta == 0)

deldtheta = cvx.Variable()
constraints.append(deldtheta == 0)
delthetas.append(deltheta)
deldthetas.append(deldtheta)

for i in range(lookahead):
    at = cvx.Variable()
    controls.append(at)

    # next time step variables
    delthetan = cvx.Variable()
    deldthetan = cvx.Variable()    
    delthetas.append(delthetan)
    deldthetas.append(deldthetan)

    lin = linearApproxAlpha(a[i], theta[i])
    #Dynamics
    constraints.append(delthetan == deltheta + (deldtheta + dtheta[i]) * dt)
    constraints.append(deldthetan == deldtheta + at * pdada * dt + deltheta * dt * pdadtheta)# lin(at, deltheta) * dt)

    #conditions on allowable control
    constraints.append(at + a[i] <= 10.0)   
    constraints.append(at + a[i] >= -10.0)
    #trust regions
    #constraints.append(delthetan <= 0.5)   
    #constraints.append(delthetan >= -0.5)
    #constraints.append(deldthetan <= 0.5)   
    #constraints.append(deldthetan >= -0.5)


    #Cost calculation  
    cost = cost + cvx.square( deltheta + theta[i] - np.pi ) #+ 0.1 * cvx.square(ut)
 

    deltheta = delthetan
    deldtheta = deldthetan


 
objective = cvx.Minimize(cost)
#print(objective)
#print(constraints)
prob = cvx.Problem(objective, constraints)
'''
sol = prob.solve(verbose=True)
print(sol)
#update by the del
theta += np.array(list(map( lambda x: x.value, delthetas))) 
dtheta += np.array(list(map( lambda x: x.value, deldthetas))) 
a += np.array(list(map( lambda x: x.value, controls)))
'''
 
for j in range(5):
    palpha.value = alpha(a, theta[:-1])
    pdada.value, pdadtheta.value = dalpha(a, theta[:-1])
    sol = prob.solve(verbose=True)
    theta += np.array(list(map( lambda x: x.value, delthetas))) 
    dtheta += np.array(list(map( lambda x: x.value, deldthetas))) 
    a += np.array(list(map( lambda x: x.value, controls)))

plt.plot(theta)
plt.plot(a)

plt.show()
