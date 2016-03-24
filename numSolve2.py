""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 Analytical solution for Geosynthetic-wrapped soil
"""

from math import *
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ

def stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3):
	"""Stress ratio q/p = q/p at failure state - stress ratio difference (a function of e_q)
	"""
	# q/p at failure state
	M_f = M*(p/p_c)**n
	# stress ratio difference
	etaDiff = a_1/(e_q+a_2)+a_3
	eta = M_f+etaDiff
	return etaDiff, eta
	
def dilationRatio(p, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2):
	"""Linear relationship between stress ratio difference and dilation ratio difference
	"""
	etaDiff, eta = stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3)
	M_f = eta-etaDiff
	K_mu = -(M-M_f)+b_1*etaDiff+b_2
	return K_mu

def plaStrainIncrements(dp, p, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2, E, v):
   """axial plastic strain increment
   """
   # ratio of mean to deviatoric stress increment
   etaDiff, eta = stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3)
   # ratio of plastic volumetric to deviator strain increment
   K_mu = dilationRatio(p, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2)
   
   # ratio of axial to radial stress increment
   dK_p  = (2.*eta+3.)/(3.-eta)
   # ratio of radial strain to axial strain increment
   K_phi = (2.*-K_mu-3)/(2.*-K_mu+6.)
   
   # generalized modulus
   D = E/(2.+K_phi*dK_p-2.*v*(1.+dK_p+K_phi))
   
   # expressions for axial strain increment
   de_a = dp/(D*K_phi*(dK_p+2.))
   # expressions for radial strain increment
   de_r = dp/(D*(dK_p+2.))
   return de_a, de_r

def plaStrains(dp, p, e_Pla_a, e_Pla_r, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2, E, v):
   de_a, de_r = plaStrainIncrements(dp, p, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2, E, v)
   e_Pla_a = de_a + e_Pla_a
   e_Pla_r = de_r + e_Pla_r
   return e_Pla_a, e_Pla_r
   
def elaStrains(dp, p, C_t, p_a, p_0, m):
   """This is the function which returns axial and radial elastic strain
   """
   p = dp +p
   e_v = C_t*((p/p_a)**m-(p_0/p_a)**m)
   return 1./3.*e_v, 1./3.*e_v
   
def analSolution(dp, p, e_q, e_Pla_a, e_Pla_r):
   """This function returns 0 when the equation of analytical solution is satisfied
   """
   # ratio of mean to deviatoric stress increment
   etaDiff, eta = stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3)
   # vertical principal stress
   s_a = (p+dp)*(2.*eta+3.)/3.
   # horizontal principal stress
   s_r = (p+dp)*(-eta+3.)/3.
   # axial and radial elastic strain
   e_Ela_a, e_Ela_r = elaStrains(dp, p, C_t, p_a, p_0, m)
   # axial and radial elastic strain
   e_Pla_a, e_Pla_r = plaStrains(dp, p, e_Pla_a, e_Pla_r, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2, E, v)
   # total strain
   e_a = e_Ela_a+e_Pla_a
   e_r = e_Ela_r+e_Pla_r
   # equation of analytical solution
   result = s_a-s_af-2.*(s_r-s_rf)*(H/B)*(1.-0.01*e_a)/((H/B+1.)*(1.-0.01*e_r))
   return result

def getStrains(dp, p, e_q, e_Pla_a, e_Pla_r):
   """This function computes axial and radial strains of geosynthetic-wrapped soil,
      as well as tension strain of the wrapping membrane
   """
   # axial and radial elastic strain
   e_Ela_a, e_Ela_r = elaStrains(dp, p, C_t, p_a, p_0, m)
   # axial and radial plastic strain
   e_Pla_a, e_Pla_r = plaStrains(dp, p, e_Pla_a, e_Pla_r, e_q, M, p_c, n, a_1, a_2, a_3, b_1, b_2, E, v)
   e_q = 2./3.*(e_Pla_a-e_Pla_r)
   # total strain
   e_a = e_Ela_a+e_Pla_a
   e_r = e_Ela_r+e_Pla_r
   # membrane tension strain
   etaDiff, eta = stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3)
   s_r = p*(-eta+3.)/3.
   rhs = (s_r-s_rf)*B*H*(1.-0.01*e_a)*(1.-0.01*e_r)
   lhs = J*(2.*B+2.*H)
   e_T = 100*rhs/lhs
   return e_a, e_r, e_Pla_a, e_Pla_r, e_q, e_T
   
def getInternalStress(p):
   """This function computes internal vertical and horizontal stress in geosynthetic-wrapped soil
   """
   # ratio of total mean stress to deviatoric stress
   etaDiff, eta = stressRatio(p, e_q, M, p_c, n, a_1, a_2, a_3)
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
GSType = 'PE'
if GSType == 'PE': E = 23529.16665
v = 0.33
J = 53*1e3

# state parameter
M = 1.35
p_c = 3.255e6
n = -0.0723

# elastic parameter
C_t = 1.372
p_a = 1.e5
m = 0.3

# plastic parameter
a_1 = -1.386
a_2 = 1.27
a_3 = 0.0346
b_1 = 1.382
b_2 = 0.2354


#######################################
##  initial and boundary conditions  ##
#######################################

# initial mean stress
p_0 = 2.6e3

# external lateral confining stress
s_rf = 0.

# min and max external vertical load
s_afMin =   4.212e3
s_afMax = 476.694e3

# initial state variables
e_q = 0; p = p_0; dp = 0
e_Pla_a = 0; e_Pla_r = 0

#############################
##  Numerical integration  ##
#############################

# get internal vertical, lateral principal stress and strains
data = {}
data['s_a'] = []; data['s_r'] = []; data['s_af'] = []
data['e_a'] = []; data['e_r'] = []; data['e_T'] = []

# perform integration with external load increment
num = 1e5
dS_af = (s_afMax-s_afMin)/num
for i in range(int(num)):
	s_af = s_afMin + dS_af*(i+1)
	dp = opt.fsolve(analSolution,dp,args=(p,e_q,e_Pla_a,e_Pla_r))
	e_a, e_r, e_Pla_a, e_Pla_r, e_q, e_T = getStrains(dp, p, e_q, e_Pla_a, e_Pla_r)
	p += dp
	if i == num or i%1000 == 0:
		s_a,s_r = getInternalStress(p)
		data['s_a'].append(s_a); data['s_r'].append(s_r); data['s_af'].append(s_af) 
		data['e_a'].append(e_a); data['e_r'].append(e_r); data['e_T'].append(e_T) 

##########################
##  Write data to file  ##
##########################

fout = file('analSolve'+GSType+'.dat','w')
for i in range(len(data['s_a'])):
	fout.write('%15.3f'%data['s_a'][i]+'%15.3f'%data['s_r'][i]+'%15.3f'%data['s_af'][i] \
	           +'%9.3f'%data['e_a'][i] +'%9.3f'%data['e_r'][i]+ '%9.3f'%data['e_T'][i]+'\n')
fout.close()
