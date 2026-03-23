# grid_tau.py
import numpy as np, csv
G = 6.67430e-11
hbar = 1.054571817e-34

def deltaE_G(m,d):
    return G * m*m / d

def tau_from(m,d):
    dE = deltaE_G(m,d)
    if dE <= 0: return np.inf
    return hbar / dE

masses = np.logspace(-25, -12, 8)   # kg
dists = np.logspace(-9, -6, 7)      # m

with open('tau_grid.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m(kg)','d(m)','DeltaE(J)','tau(s)','gamma(1/s)'])
    for m in masses:
        for d in dists:
            dE = deltaE_G(m,d)
            tau = tau_from(m,d)
            gamma = 0.0 if np.isinf(tau) else 1.0/tau
            writer.writerow([m,d,dE,tau,gamma])
print("Saved tau_grid.csv")
