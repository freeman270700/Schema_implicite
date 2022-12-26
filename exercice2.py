
import matplotlib.pyplot as plt
import numpy as np
from numpy import nbytes, zeros,array,dot,linspace, linalg
from math import *
import matplotlib.pyplot as plt

S = 50
K = 50
r = 0.05
sigma = 0.2
T = 3

M = 150
N = 200
Szero = 0
Smax = 150

solution_mesh = np.zeros((N+1,M+1))
Smesh = np.linspace(0, Smax, M+1)
Tmesh = np.linspace(T, 0, N+1)
dt = T/N
solution_mesh[0,:] = np.maximum(K-Smesh, 0)
solution_mesh[:,0] = K*np.exp(-r*(T-Tmesh))
solution_mesh[:,M] = 0

def A(i):
    return 0.5 * dt*(r*i-sigma**2*i**2)

def B(i):
    return 1 + (sigma**2*i**2 + r)*dt

def C(i):
    return -0.5*dt*(sigma**2*i**2+r*i)

Acoeffs = np.zeros(M+1)
Bcoeffs = np.zeros(M+1)
Ccoeffs = np.zeros(M+1)
for i in range(M+1):
    Acoeffs[i] = A(i-1)
    Bcoeffs[i] = B(i-1)
    Ccoeffs[i] = C(i-1)

Tri = np.diag(Acoeffs[1:], -1) + np.diag(Bcoeffs) + np.diag(Ccoeffs[:-1], 1)
Tri_Inv = np.linalg.inv(Tri)

for j in range(N):
    temp = np.zeros(M+1)
    temp[0] = A(0)*solution_mesh[j+1,0]
    temp[-1] = C(M)*solution_mesh[j+1,M]
    RHS = solution_mesh[j,:] - temp
    temp = Tri_Inv.dot(RHS)
    solution_mesh[j+1,1:-1] = temp[1:-1]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
Smesh, Tmesh = np.meshgrid(Smesh, Tmesh)
ax.plot_surface(Smesh, Tmesh, solution_mesh)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V(t,s)', rotation=0)

# Graphique
fig = plt.figure(2)
plt.plot(Tmesh, solution_mesh[:,0], Tmesh, solution_mesh[:,9], Tmesh, solution_mesh[:,49], Tmesh, solution_mesh[:,99], Tmesh, solution_mesh[:,149])
plt.xlabel('Temps')
plt.ylabel('Sous-jacent(S)')
plt.legend(['0','50','100','150'])
plt.show()
