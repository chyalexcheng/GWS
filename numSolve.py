""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 Analytical solution for Geosynthetic-wrapped soil
"""

from math import *
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ

def integrdEp_a(p, E, v, M, p_c, n, M_c, b):
   """This is the function inside the integral of axial plastic strain
   """
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
   """This is the function inside the integral of radial plastic strain
   """
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

def e_Ela(p):
   """This is the function which returns axial and radial elastic strain
   """
   e_v = C_t*((p/p_a)**m-(p_0/p_a)**m)
   eta = M*(p/p_c)**n
   e_a = ((2.*eta-6.)*v+2.*eta+3.)/(9.*(1.-2.*v))*e_v
   e_r = -((eta+6.)*v+eta-3.)/(9.*(1.-2.*v))*e_v
   return e_a, e_r

def coefficient(p):
   """This function returns key parameters in plastic regime calculation
   """
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

def e_Pla_a(p):
   """This function computes the integral of axial plastic strain increment
   """
   (result,error) = integ.quad(integrdEp_a, p_0, p, args=(E, v, M, p_c, n, M_c, b))
   return result

def e_Pla_r(p):
   """This function computes the integral of radial plastic strain increment
   """
   (result,error) = integ.quad(integrdEp_r, p_0, p, args=(E, v, M, p_c, n, M_c, b))
   return result
   
def analSolution(p):
   """This function returns 0 when the equation of analytical solution is satisfied
   """
   # ratio of total mean stress to deviatoric stress
   eta = M*(p/p_c)**n
   # vertical principal stress
   s_a = p*(2.*eta+3.)/3.
   # horizontal principal stress
   s_r = p*(-eta+3.)/3.
   # axial and radial elastic strain
   e_Ela_a, e_Ela_r = e_Ela(p)
   # total axial strain
   e_a = e_Ela_a+e_Pla_a(p)
   # total radial strain
   e_r = e_Ela_r+e_Pla_r(p)
   # equation of analytical solution
   result = s_a-s_af-2.*(s_r-s_rf)*(H/B)*(1.-0.01*e_a)/((H/B+1.)*(1.-0.01*e_r))
   return result

def getStrains(p):
   """This function computes axial and radial strains of geosynthetic-wrapped soil,
      as well as tension strain of the wrapping membrane
   """
   # axial and radial elastic strain
   e_Ela_a, e_Ela_r = e_Ela(p)
   # total axial strain
   e_a = e_Ela_a+e_Pla_a(p)
   # total radial strain
   e_r = e_Ela_r+e_Pla_r(p)
   # membrane tension strain
   eta = M*(p/p_c)**n
   s_r = p*(-eta+3.)/3.
   rhs = (s_r-s_rf)*B*H*(1.-0.01*e_a)*(1.-0.01*e_r)
   lhs = J*(2.*B+2.*H)
   e_T = 100*rhs/lhs
   return e_a, e_r, e_T
   
def getInternalStress(p):
   """This function computes internal vertical and horizontal stress in geosynthetic-wrapped soil
   """
   # ratio of total mean stress to deviatoric stress
   eta = M*(p/p_c)**n
   # vertical principal stress
   s_a = p*(2.*eta+3.)/3.
   # horizontal principal stress
   s_r = p*(-eta+3.)/3.
   return s_a, s_r

#########################
##  Define parameters  ##
#########################

# size of geosynthetic wrapped soil
H = 0.08
B = 0.4

# materal constants
E = 23529.16665
v = 0.33
J = 53*1e3

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

#######################################
##  initial and boundary conditions  ##
#######################################

# initial mean stress
p_0 = 2.036e3

# external lateral confining stress
s_rf = 0.

# loading history
ls_af = np.array([4.212207707,12.22138428,43.2208806,85.46887137,132.4727983,
       181.8801554,234.2676809,282.4550162,328.0113041,369.6877088,
       413.6410706,448.1422977,476.6944715,465.6414952,429.3896047,
       419.0090083,376.1143861,348.4557448,341.6402818,325.9653198,
       291.4893836])*1e3
ls_af = list(ls_af)

####################################################################
##  get analytical solutions with known external vertical stress  ##
####################################################################

data = {}
# get a list of mean stress
data['p'] = []
for i in range(len(ls_af)):
    s_af = ls_af[i]
    data['p'].append(opt.fsolve(analSolution,p_0))

# get internal vertical, lateral principal stress and strains
data['s_a'] = []; data['s_r'] = []; data['e_v'] = []
data['e_a'] = []; data['e_r'] = []; data['e_T'] = []
for i in range(len(data['p'])):
	s_a,s_r = getInternalStress(data['p'][i])
	e_a,e_r,e_T = getStrains(data['p'][i])
	data['s_a'].append(s_a); data['s_r'].append(e_r) 
	data['e_a'].append(e_a); data['e_r'].append(e_r); data['e_T'].append(e_T) 

""" Turn on this to do debugging
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
"""
