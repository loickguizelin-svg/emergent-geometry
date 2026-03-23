# diosi_penrose.py
import numpy as np

G = 6.67430e-11
hbar = 1.054571817e-34

def delta_EG_spheres(m, R, d):
    """
    Approximate gravitational self-energy difference for two identical uniform spheres
    of mass m and radius R separated by distance d (center-to-center).
    Valid for d >= 2R (non overlapping) as rough estimate: DeltaE ~ G m^2 / d
    For overlapping or close spheres a more precise integral is needed.
    """
    if d <= 2*R:
        # overlapping regime: use approximate constant ~ G m^2 / R
        return G * m**2 / (2*R)
    return G * m**2 / d

def tau_from_deltaE(deltaE):
    if deltaE <= 0:
        return np.inf
    return hbar / deltaE

# convenience wrapper
def tau_for_spheres(m, R, d):
    dE = delta_EG_spheres(m, R, d)
    tau = tau_from_deltaE(dE)
    return dE, tau

# example usage:
# m = 1e-15  # kg
# R = 1e-6   # m
# d = 1e-6   # m
# dE, tau = tau_for_spheres(m,R,d)
# print("DeltaE_G:", dE, "J ; tau:", tau, "s")
