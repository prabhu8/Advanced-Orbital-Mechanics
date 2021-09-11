import numpy as np
import math 
import matplotlib.pyplot as plt

# Angle Manupulation 
# Wrapto2pi
def wrapto2pi(angle):
    return angle % (2*np.pi)


#  Semi-Major axis 
# With r_p and r_a (distance to periapsis and apoapsis)
def a_rp_ra(r_p, r_a):
    return 0.5*(r_a+r_p)


# With miu and Eps (Gravitational parameter and Specific Energy)
def a_miu_Eps(miu, Eps):
    return -miu/2/Eps

##### Eccentricity #####
# With r_p and a (distance to periapsis and Semi-major axis)
def e_rp_a(r_p, a):
    return 1-r_p/a

def e_a_p(a, p):
    return np.sqrt(1-p/a)

##### Semi-Latus Rectum #####
# With a and e (Semi-major axis and eccentricity)
def p_a_e(a, e):
    return a*(1-e**2)

def p_miu_h(miu, h):
    return h**2/miu

##### Specific Angular Momentum #####
# With miu and p (Mass constant and Semi-Latus Rectum)
def h_miu_p(miu, p):
    return np.sqrt(miu*p)

# With r_vec and v_vec (Distance and velocity)
def h_rvec_vvec(r_vec,v_vec):
    return np.linalg.norm(np.cross(r_vec,v_vec))

##### Specific energy #####
# With miu and a (Mass constant and Semi-Major Axis)
def Eps_miu_a(miu,a):
    return -miu/2/a

def Eps_miu_r_v(miu, r, v):
    return v**2/2 -miu/r

##### Period #####
# With miu and a (Mass constant and Semi-Major Axis)
def Per_miu_a(miu,a):
    return 2*np.pi*np.sqrt(a**3/miu)

##### Eccentric Anomaly #####
# With miu and a (Mass constant and Semi-Major Axis)
def E_thst(thst,e):
    return 2*math.atan2(math.tan(thst/2),math.sqrt((1+e)/(1-e)))

# With just M (mean Anomaly)
def E_M(M, e, acc = 10^-12, **kwargs):
    E_n = M
    for n in range(100):    
        E_n1 = E_n - (E_n - e*math.sin(E_n) - M)/(1-e*math.cos(E_n))
        error = abs(E_n1 - E_n)
        
        if error < acc:
            break
        
        E_n = E_n1
    
    if 'n_count' in kwargs and kwargs["n_count"] == True:
        return [E_n1, n+1]
    else:
        return E_n1
        

##### Mean Anomaly #####
# With miu and a (Mass constant and Semi-Major Axis)
def M_E_e(E,e):
    return E-e*math.sin(E)



##### Keplar radial Distance #####
def keplar_r(p,e,th_st):
    return p/(1+e*np.cos(th_st))

##### Speed #####
def v_a_r(miu, a,r):
    return np.sqrt(2*(miu/r - miu/2/a))


##### True Anomaly #####
def thst_rvec_vvec_p_e(r_vec,v_vec,p,e):
  return np.sign(np.dot(v_vec,r_vec)) * math.acos(1/e*(p/np.linalg.norm(r_vec) - 1))


def thst_E_e(E, e):
    return 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))

##### Change in True Anomaly #####
def delthst_r1vec_r2vec_hunit(r1_vec,r2_vec,h_unit):
    return np.sign(h_unit[-1])*math.acos( np.dot(r1_vec,r2_vec)/np.linalg.norm(r1_vec)/np.linalg.norm(r2_vec) )


##### Anomaly #####
def theta_runit_thunit(r_unit,h_unit):
  return math.atan2(r_unit[-1],h_unit[-1])


##### Flight Path Angle #####
def gamma_h_r_v_thst(h,r,v,thst):
  if wrapto2pi(thst) > np.pi:
    return -math.acos(h/r/v)
  else:
    return math.acos(h/r/v)


##### Inclination #####
def i_hunit(h_unit_ECI):
  return math.acos(h_unit_ECI[-1])


##### Right Ascension #####
def RAAN_hunit(h_unit_ECI):
  return math.atan2(h_unit_ECI[0],-h_unit_ECI[1])


##### Argument of Periapsis #####
def omega_thst_theta(thst,theta):
  return theta - thst


##### f and g function #####
def f_r2_p_delthst(r2,p,delthst):
  return 1 - r2/p*(1-math.cos(delthst))

def g_miu_r1_r2_p_delthst(miu,r1,r2,p,delthst):
  return r1*r2/np.sqrt(miu*p)*math.sin(delthst)

def fdot_miu_r1vec_v1vec_p_delthst(miu, r1_vec,v1_vec,p,delthst):
    r1 = np.norm(r1_vec)
    return np.dot(r1_vec,v1_vec)/p/np.linalg.norm(r1_vec)*(1-math.cos(delthst)) -1/r1*np.sqrt(miu/p)*math.sin(delthst)

def gdot_r1_p_delthst(r1,p,delthst):
    return 1-r1/p*(1-math.cos(delthst))



##### Change of Frames #####
# Rotation around axis 1
def DCM_1(beta):
    return np.array([[1,0,0], \
                     [0, np.cos(beta), -np.sin(beta)] , \
                    [0, np.sin(beta), np.cos(beta)]])

# Rotation around axis 3
def DCM_3(beta):
    return np.array([[np.cos(beta), -np.sin(beta), 0] , \
                    [np.sin(beta), np.cos(beta), 0],\
                    [0,0,1]])

# Satellite to Perifocal 
def sat2per(th_st):
    return DCM_3(th_st)

# ECI to Satellite frame
def sat2ECI(Omega, i, theta):
    return DCM_3(Omega) @ DCM_1(i) @ DCM_3(theta)



#### Radial to inertial ####
def rad_2_cart(r,angle):
    x = r*np.sin(angle)
    y = r*np.cos(angle)
    return x,y

##### Plot Functions #####
# 3d axis
def lims3dplot(xmax,xmin,ymax,ymin,zmax,zmin):
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max() 
    mid_x = (xmax+xmin)/2
    mid_y = (ymax+ymin)/2
    mid_z = (zmax+zmin)/2
    
    mids = np.array([mid_x,mid_y,mid_z])
    upper_lim = np.array([-max_range, -max_range, -max_range]) + mids
    lower_lim = np.array([max_range, max_range, max_range]) + mids
    return lower_lim, upper_lim


#### Plot 2d ####
def plot_2d(a, e, thst, ax, vis = '-'):
    r = a*(1-e**2)/(1+e*np.cos(thst))
    x = r*np.cos(thst)
    y = r*np.sin(thst)
    
    ax.plot(x,y, vis)


  