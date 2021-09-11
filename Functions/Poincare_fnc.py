import numpy as np

#### Dimensionalize ####
def L_2_dim(length, l_star):
    return length * l_star

def M_2_dim(mass, m_star):
    return mass * m_star

def t_2_dim(time, t_star):
    return time*t_star

#### Lagrange Points ####
def L2_Newton(miu, gamma_n, acc = 10**-8):
    
    for n in range(100):
        f_n = gamma_n**5 + gamma_n**4*(3-miu) + gamma_n**3*(3-2*miu) - miu*gamma_n**2 - 2*gamma_n*miu - miu
        f_n_p = 5*gamma_n**4 + 4*gamma_n**3*(3-miu) + 3*gamma_n**2*(3-2*miu) - 2*miu*gamma_n - 2*miu 
        
        gamma_n1 = gamma_n - f_n/f_n_p        
        if abs(gamma_n1-gamma_n) < acc:
            break
        
        gamma_n = gamma_n1

    L2_nondim = 1 - miu + gamma_n
    return gamma_n1, L2_nondim


def L1_Newton(miu, gamma_n, acc = 10**-8):
    
    for n in range(100):
        f_n = -gamma_n**5 + gamma_n**4*(3-miu) + gamma_n**3*(2*miu-3) + miu*gamma_n**2 - 2*gamma_n*miu + miu
        f_n_p = -5*gamma_n**4 + 4*gamma_n**3*(3-miu) + 3*gamma_n**2*(2*miu-3) + 2*miu*gamma_n - 2*miu 
        
        gamma_n1 = gamma_n - f_n/f_n_p        
        if abs(gamma_n1-gamma_n) < acc:
            break
        
        gamma_n = gamma_n1

    L1_nondim = 1 - miu - gamma_n
    return gamma_n1, L1_nondim

def L3_Newton(miu, gamma_n, acc = 10**-8):
    
    for n in range(100):
        f_n = gamma_n**5 + gamma_n**4*(miu+2) + gamma_n**3*(2*miu+1) + gamma_n**2*(miu-1) + gamma_n*(2*miu-2) + miu - 1

        f_n_p = 5*gamma_n**4 + 4*gamma_n**3*(miu+2) + 3*gamma_n**2*(2*miu+1) + 2*miu*gamma_n*(miu-1) + 2*miu - 2
        
        gamma_n1 = gamma_n - f_n/f_n_p        
        if abs(gamma_n1-gamma_n) < acc:
            break
        
        gamma_n = gamma_n1

    L3_nondim = -gamma_n - miu
    return gamma_n1, L3_nondim

#### d and r ####
def d_n_r(x, y, z, miu):
    d = np.sqrt((x+miu)**2 + y**2 + z**2)
    r = np.sqrt((x+miu-1)**2 + y**2 + z**2)
    return d, r


#### Jacobi Constant ####
def Jacobi_const(x, y, z, v, miu):
    d, r = d_n_r(x, y, z, miu)
    return (x**2 + y**2) + 2*(1-miu)/d + 2*miu/r - v**2

#### ZVC Curve ####
