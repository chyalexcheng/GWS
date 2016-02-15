from sympy import *

# material constants
K_p, K_phi, dK_p = symbols('K_p K_phi dK_p')
E, v = symbols('E v')
# generalized modulus
D = E/(2.+K_phi*dK_p-2.*v*(1.+dK_p+K_phi))
# expression for axial and radial strain increment
de_a, de_r = symbols('de_a de_r')
de_a = 1./(D*K_phi*(K_p+2.))
de_r = 1./(D*(K_p+2.))
# simply expression
de_a = de_a.simplify()
de_r = de_r.simplify()

# use stress-dilation relation to compute materal constants
eta, K_mu, dEta = symbols('eta K_mu dEta')
de_a = de_a.subs([(K_p,  (2*eta+3)/(3-eta)), 
                 (K_phi,(2*K_mu-3)/(2*K_mu+6)),
                 (dK_p, (2*dEta+3)/(3-dEta))])
de_r = de_r.subs([(K_p,  (2*eta+3)/(3-eta)), 
                 (K_phi,(2*K_mu-3)/(2*K_mu+6)),
                 (dK_p, (2*dEta+3)/(3-dEta))])
# do cancellatin on expression
de_a = de_a.cancel()
de_r = de_r.cancel()

M, p, p_c, n, M_c, b = symbols('M, p, p_c, n, M_c, b')
de_a = de_a.subs([(eta,  M*(p/p_c)**n),
                 (K_mu, b-(M_c**2-dEta**2)/(2*dEta)),
                 (dEta, M*(1+n)*(p/p_c)**n)])
de_r = de_a.subs([(eta,  M*(p/p_c)**n),
                 (K_mu, b-(M_c**2-dEta**2)/(2*dEta)),
                 (dEta, M*(1+n)*(p/p_c)**n)])

# do cancellatin on expression
de_a = de_a.cancel()
de_r = de_r.cancel()

# integrate de_a and de_r
e_a = integrate(de_a,p)
e_r = integrate(de_r,p)
