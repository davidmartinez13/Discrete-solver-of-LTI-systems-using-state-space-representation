import numpy as np
import matplotlib.pyplot as plt

def solver(A,B,C,u,dt,X0):

    out_samples = len(u)
    #preparing arrays for output
    xout = np.zeros((2,out_samples))
    yout = np.zeros((out_samples))
    #time of simulation
    stoptime = (out_samples - 1) * dt
    tout = np.linspace(0.0, stoptime, num=out_samples)
    #solving state space with matrix exponential approach 
    # see pages 52-54: http://eceweb1.rutgers.edu/~gajic/solmanual/slides/chapter8_DIS.pdf
    for k in range(0, out_samples-1):
        for m in range(0,k-1):
            A_dot_X0=np.dot(np.linalg.matrix_power(A,k),X0)#dot product of A with X0
            B_dot_u=np.dot(B,u[m])#dot product of B with u
            Aexp_dot_Bu =np.dot(np.linalg.matrix_power(A,k-m-1), B_dot_u)#dot product matrix pow with Bu
            #dot products with C
            dot1=np.dot(C,A_dot_X0)
            dot2=np.dot( C ,Aexp_dot_Bu)
            #final sums and construction of output arrays
            xout[:,k]=(A_dot_X0 + Aexp_dot_Bu).T
            yout[k]=dot1+ dot2
             
    return {'tout':tout, 'yout':yout, 'xout':xout}#return arrays of time, outputy and output x as dictionary

#matrices
A = np.array([[0.0,1.0],[-1.3,-1.0]])
B = np.array([[0.0],[0.5]])
C = np.array([1,0])
# D = np.array([0.0])
u= np.ones(100)*8
u[50:]= 0
X0=np.array([[50],[50]])
dt=1
output=solver(A,B,C,u,dt,X0)

#ploting figure
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(output['tout'],output['yout'],'b',linewidth=1)
plt.ylabel('y')
plt.xlabel('t')
plt.legend(['Signal'],loc='best')

plt.subplot(2,2,2)
plt.plot(output['tout'],u,'r',linewidth=1)
plt.ylabel('u')
plt.xlabel('t')
plt.legend(['Input u'],loc='best')

plt.subplot(2,2,3)
plt.plot(output['xout'][0,:],output['xout'][1,:],'m',linewidth=0.5)
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(['x1 in terms of x2'],loc='best')
plt.show()