import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
def Jacobian(psi_hat,zeta_hat):
    psi_x,psi_y = np.fft.irfft2(1j*k*psi_hat,s=(Nk,Nk)),np.fft.irfft2(1j*l*psi_hat,s=(Nk,Nk))
    zeta_x,zeta_y = np.fft.irfft2(1j*k*zeta_hat,s=(Nk,Nk)),np.fft.irfft2(1j*l*zeta_hat,s=(Nk,Nk))
    return psi_x*zeta_y-psi_y*zeta_x

def psi_hat_from_zeta_hat(zeta_hat):
    return np.divide(-zeta_hat,kappa_sq,out = np.zeros_like(zeta_hat),where=kappa_sq!=0) # don't divide when k = l = 0

def euler(psi_hat,zeta_hat,F_hat,dt,total_time,steps):
    for i in range(steps):
        zeta_hat += dt*(F_hat-nu*kappa_sq*zeta_hat-np.fft.rfft2(Jacobian(psi_hat,zeta_hat)))
        psi_hat = psi_hat_from_zeta_hat(zeta_hat)
    total_time += steps*dt
    return zeta_hat,psi_hat,total_time

def rk4_values(psi_hat,zeta_hat,F_hat,dt):
    k1 = F_hat-nu*kappa_sq*zeta_hat-np.fft.rfft2(Jacobian(psi_hat,zeta_hat))
    k2 = F_hat-nu*kappa_sq*(zeta_hat+dt*k1/2)-np.fft.rfft2(Jacobian(psi_hat_from_zeta_hat(zeta_hat+dt*k1/2),zeta_hat+dt*k1/2))
    k3 = F_hat-nu*kappa_sq*(zeta_hat+dt*k2/2)-np.fft.rfft2(Jacobian(psi_hat_from_zeta_hat(zeta_hat+dt*k2/2),zeta_hat+dt*k2/2))
    k4 = F_hat-nu*kappa_sq*(zeta_hat+dt*k3)-np.fft.rfft2(Jacobian(psi_hat_from_zeta_hat(zeta_hat+dt*k3),zeta_hat+dt*k3))
    return k1,k2,k3,k4

def rk4(psi_hat,zeta_hat,F_hat,dt,total_time,steps):
    for i in range(steps):
        k1,k2,k3,k4 = rk4_values(psi_hat,zeta_hat,F_hat,dt)
        zeta_hat += dt/6*(k1+2*k2+2*k3+k4)
        psi_hat = psi_hat_from_zeta_hat(zeta_hat)
    total_time += steps*dt
    return zeta_hat,psi_hat,total_time
########################################################################################################################

L = 1
nu = 0.005
Nk = 128
x,y = np.meshgrid(np.linspace(0,1,Nk,endpoint=False),np.linspace(0,1,Nk,endpoint=False))

# frequencies
k,l = np.meshgrid(np.fft.rfftfreq(Nk,L/Nk),np.fft.fftfreq(Nk,L/Nk))
k,l = 2*np.pi*k,2*np.pi*l
kappa_sq = k**2+l**2

dt = 0.00001
total_time = 0

# initial condition
psi = np.sin(2*np.pi*x)+np.cos(4*np.pi*y)+np.sin(6*np.pi*x)*np.cos(2*np.pi*y)
psi_hat = np.fft.rfft2(psi)
zeta_hat = -kappa_sq*psi_hat

# forcing
F = np.cos(16*np.pi*x)
# F = np.zeros_like(x)
F_hat = np.fft.rfft2(F)

plt.figure()
plt.contourf(x,y,np.fft.irfft2(zeta_hat,s=(Nk,Nk)))
plt.colorbar()
plt.title('t = %1.5f' %total_time)
plt.pause(0.05)

# time stepping
for t in range(30):
    zeta_hat,psi_hat,total_time = euler(psi_hat,zeta_hat,F_hat,dt,total_time,steps = 150)
    plt.clf()
    plt.contourf(x,y,np.fft.irfft2(zeta_hat,s=(Nk,Nk)))
    plt.colorbar()
    plt.title('t = %1.5f' %total_time)
    plt.pause(0.05)

