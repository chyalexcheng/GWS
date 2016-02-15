import math
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ

def integrdEp_a(p, E, v, M, p_c, n, M_c, b):
   # ratio of mean to deviatoric stress increment
   dEta = M*(1.+n)*(p/p_c)**n
   # ratio of plastic volumetric to deviator strain increment
   K_mu = (M_c**2.-dEta**2.)/(2.*dEta)+b
   
   # ratio of axial to radial stress increment
   dK_p  = (2.*dEta+3.)/(3.-dEta)
   # ratio of radial strain to axial strain increment
   K_phi = (2.*-K_mu-3)/(2.*-K_mu+6.)
   
   # generalized modulus
   D = E/(2.+K_phi*dK_p-2.*v*(1.+dK_p+K_phi))
   
   # expressions for axial strain increment
   de_a = 1./(D*K_phi*(dK_p+2.))
   return de_a

def integrdEp_r(p, E, v, M, p_c, n, M_c, b):
   # ratio of mean to deviatoric stress increment
   dEta = M*(1.+n)*(p/p_c)**n
   # ratio of plastic volumetric to deviator strain increment
   K_mu = (M_c**2.-dEta**2.)/(2.*dEta)+b
   
   # ratio of axial to radial stress increment
   dK_p  = (2.*dEta+3.)/(3.-dEta)
   # ratio of radial strain to axial strain increment
   K_phi = (2.*-K_mu-3)/(2.*-K_mu+6.)
   
   # generalized modulus
   D = E/(2.+K_phi*dK_p-2.*v*(1.+dK_p+K_phi))
   
   # expressions for radial strain increment
   de_r = 1./(D*(dK_p+2.))
   return de_r

def coefficient(p, E, v, M, p_c, n, M_c, b):
   # ratio of mean to deviatoric stress increment
   dEta = M*(1.+n)*(p/p_c)**n
   # ratio of plastic volumetric to deviator strain increment
   K_mu = (M_c**2.-dEta**2.)/(2.*dEta)+b
   
   # ratio of axial to radial stress increment
   dK_p  = (2.*dEta+3.)/(3.-dEta)
   # ratio of radial strain to axial strain increment
   K_phi = (2.*-K_mu-3)/(2.*-K_mu+6.)
   
   # generalized modulus
   D = E/(2.+K_phi*dK_p-2.*v*(1.+dK_p+K_phi))
   return "dEta: %2.2f"%dEta,"K_mu: %2.2f"%K_mu,"dK_p: %2.2f"%dK_p,"K_phi: %2.2f"%K_phi,"D: %2.2f"%D

def e_Ela(p, C_t, p_a, p_0, m, M, p_c, n, v):
	e_v = C_t*((p/p_a)**m-(p_0/p_a)**m)
	eta = M*(p/p_c)**n
	e_a = ((2.*eta-6.)*v+2.*eta+3.)/(9.*(1.-2.*v))*e_v
	e_r = -((eta+6.)*v+eta-3.)/(9.*(1.-2.*v))*e_v
	return e_a, e_r

# materal constants
E = 23529.16665
v = 0.33

# state parameter
M = 1.35
p_c = 3.255e6
n = -0.0723

# elastic parameter
C_t = 1.34
p_a = 1.e5
m = 0.3

# plastic parameter
M_c = sqrt(1.837475608)
b = 0.5655

# initial mean stress
p_0 = 2.036e3

########################################################
##  Check elastic and plastic strain given p history  ##
########################################################

# mean stress history
p = [p_0,7369.25026,25328.29263,49400.54306,76159.72078,104360.6706,134186.9436,
     162015.3376,188604.7941,213458.3416,239369.3651,260588.737,278776.5192,
     277316.5256,262834.2942,259877.6282,239984.4308]

# compute elastic e_a and e_r history
e_Ela_a, e_Ela_r = [],[]
for i in range(len(p)):
	e_a, e_r = e_Ela(p[i], C_t, p_a, p_0, m, M, p_c, n, v)
	e_Ela_a.append(e_a)
	e_Ela_r.append(e_r)

# compute plastic e_a and e_r history
e_Pla_a, e_Pla_r = [],[]
for i in range(len(p)):
   e_Pla_a.append(integ.quad(integrdEp_a, p_0, p[i], args=(E, v, M, p_c, n, M_c, b))[0])
   e_Pla_r.append(integ.quad(integrdEp_r, p_0, p[i], args=(E, v, M, p_c, n, M_c, b))[0])
