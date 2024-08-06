import numpy as np

# nrt defect production model for dose calculation
def nrtdamage(Td, Ed):
    if Td <= Ed:
        return 0.0
    elif Td <= 2*Ed/0.8:
        return 1.0
    else:
        return 0.8*Td/(2*Ed)

def quickdamage(Td, Ed):
    return 0.8*Td/(2*Ed)

# lindhard stopping model
class Lindhard:
    def __init__(self, A1, A2, Z1, Z2):
        self.A1 = A1
        self.A2 = A2
        self.Z1 = Z1
        self.Z2 = Z2
       
        self._a0  = 5.29177e-11 # bohr radius in m
        self._e0  = 5.52635e7   # vacuum permittivity in e^2/(eV*m)
       
        self._k   = 0.1337 * Z1**(1/6.) * (Z1/A1)**(1/2.)
        self._a   = (9.*np.pi*np.pi/128.)**(1/3.) * self._a0 * (Z1**(2/3.) + Z2**(2/3.))**(-1/2.)
   
    def _ge(self, e):
        return 3.4008* e**(1/6.) + 0.40244* e**(3/4.) + e
   
    def __call__(self, E0):
        eps = E0*self.A2/(self.A1+self.A2) * 4*np.pi*self._e0*self._a/(self.Z1*self.Z2)
        return E0/(1 + self._k*self._ge(eps))
   

def example ():
    # defining parameters for damage model
    tdpars = {}
    tdpars["W"]  = [183.84, 183.84, 74, 74]
    tdpars["Zr"] = [91.224, 91.224, 40, 40]
    tdpars["Cu"] = [63.546, 63.546, 29, 29]

    tdmodelW  = Lindhard(*tdpars["W"])
    tdmodelZr = Lindhard(*tdpars["Zr"])
    tdmodelCu = Lindhard(*tdpars["Cu"])

    # damage energy correction due to 10 eV cutoff in simulation
    dEW  = 10-tdmodelW(10.)
    dEZr = 10-tdmodelZr(10.)
    dECu = 10-tdmodelCu(10.)

    # example
    Ed = 10000.0
    Td = tdmodelW(Ed) + dE
