import numpy as np

#### Dimensionalize ####
def L_2_dim(length, l_star):
    return length * l_star

def M_2_dim(mass, m_star):
    return mass * m_star

def t_2_dim(time, t_star):
    return time*t_star

#### d and r ####
def d_n_r(x, y, z, miu):
    d = np.sqrt((x+miu)**2 + y**2 + z**2)
    r = np.sqrt((x+miu-1)**2 + y**2 + z**2)
    return d, r

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

# def colin_Lagrange(xi0):
def L4_L5(miu, x, acc = 10**-8):
    x = x.reshape(2,1)
    
    for n in range(100):
        d, r = d_n_r(x[0], x[1], 0, miu)
        df1x = -(x[1]**2 - 2*(x[0] + miu)**2)*(1-miu)/d**5 - miu*(x[1]-2*(x[0]+miu-1)**2)/r**5 + 1
        df2y = -(-2*x[1]**2 + (x[0]+miu)**2)*(1-miu)/d**5 - miu*(-2*x[1]**2 + (x[0]+miu-1)**2)/r**5 + 1
        dfxy = 3*x[1]*(1-miu)*(x[0]+miu)/d**5 + 3*miu*x[1]*(x[0]-1-miu)/r**5
        
        FX_n = np.array([-(1-miu)*(x[0]+miu)/d**3 - miu*(x[0]-1+miu)/r**3 + x[0],
                        -(1-miu)*x[1]/d**3 - miu*x[1]/r**3 + x[1]]).reshape(2,1)
        J_n = np.array([df1x, dfxy, dfxy, df2y]).reshape(2,2)

        
        FX_n1 = x - np.linalg.inv(J_n) @ FX_n
        if np.max(abs(x - FX_n1)) < acc:
            return FX_n1  
        x = FX_n1        

    return FX_n1
        

#### Jacobi Constant ####
def Jacobi_const(x, y, z, v, miu):
    d, r = d_n_r(x, y, z, miu)
    
    return (x**2 + y**2) + 2*(1-miu)/d + 2*miu/r - v**2


#### Potentials ####
def U_ii(x, y, z, miu):
    d, r = d_n_r(x, y, z, miu)
    
    U_xx = 1 - (1-miu)/d**3 - miu/r**3 + 3*(1-miu)*(x+miu)**2/d**5 + 3*miu*(x-1+miu)**2/r**5
    U_yy = 1 - (1-miu)/d**3 - miu/r**3 + 3*(1-miu)*y**2/d**5 + 3*miu*y**2/r**5
    U_zz = -(1-miu)/d**3 - miu/r**3 + 3*(1-miu)*z**2/d**5 + 3*miu*z**2/r**5
    U_xy = 3*(1-miu)*(x+miu)*y/d**5 + 3*miu*(x-1+miu)*y/r**5
    U_xz = 3*(1-miu)*(x+miu)*z/d**5 + 3*miu*(x-1+miu)*z/r**5
    U_yz = 3*(1-miu)*y*z/d**5 + 3*miu*y*z/r**5
    
    return U_xx, U_yy, U_zz, U_xy, U_xz, U_yz


#### CRBP ####
def cr3bp_df(t, x, miu):
    dx = np.zeros((6,))
    d, r = d_n_r(x[0], x[1], x[2], miu)
    
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = 2*x[4] + x[0] - (1-miu)*(x[0]+miu)/d**3 - miu*(x[0]-1+miu)/r**3
    dx[4] = -2*x[3] + x[1] - (1-miu)*x[1]/d**3 - miu*x[1]/r**3
    dx[5] = -(1-miu)*x[2]/d**3 - miu*x[2]/r**3
    
    return dx


#### Lagrange Variational Equation ####
def Lagrange_var_df(t, x, miu, U_xx, U_yy, U_zz, U_xy):
    dx = np.zeros((6,))
    
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = 2*x[4] + U_xx*x[0] + U_xy*x[1]
    dx[4] = -2*x[3] + U_xy*x[0] + U_yy*x[1]
    dx[5] = U_zz*x[2]
    
    return dx
