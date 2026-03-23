# plot_tau_grid.py
import numpy as np, csv, matplotlib.pyplot as plt, seaborn as sns, math, os
sns.set()
plt.rcParams['figure.figsize'] = (7,5)

# lire tau_grid.csv (format: m,d,DeltaE,tau,gamma)
rows = []
with open('tau_grid.csv','r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        m = float(r[0]); d = float(r[1]); dE = float(r[2]); tau = float(r[3]); gamma = float(r[4])
        rows.append((m,d,dE,tau,gamma))
rows = np.array(rows, dtype=object)

# extraire grilles uniques
m_vals = np.unique(rows[:,0].astype(float))
d_vals = np.unique(rows[:,1].astype(float))
m_vals.sort(); d_vals.sort()

# construire matrices log10(tau) et log10(gamma)
M = len(m_vals); D = len(d_vals)
logtau = np.zeros((M,D))
loggamma = np.zeros((M,D))
for i,m in enumerate(m_vals):
    for j,d in enumerate(d_vals):
        mask = (rows[:,0].astype(float)==m) & (rows[:,1].astype(float)==d)
        if np.any(mask):
            tau = float(rows[mask][0][3])
            gamma = float(rows[mask][0][4])
            logtau[i,j] = math.log10(tau) if np.isfinite(tau) and tau>0 else np.nan
            loggamma[i,j] = math.log10(gamma) if np.isfinite(gamma) and gamma>0 else np.nan
        else:
            logtau[i,j] = np.nan
            loggamma[i,j] = np.nan

# plot log10(tau)
plt.figure()
sns.heatmap(logtau, xticklabels=[f"{d:.0e}" for d in d_vals], yticklabels=[f"{m:.0e}" for m in m_vals],
            cmap='magma', annot=True, fmt=".2f", cbar_kws={'label':'log10(tau [s])'})
plt.xlabel('d (m)'); plt.ylabel('m (kg)')
plt.title('log10(tau) grid')
plt.tight_layout()
os.makedirs('tau_plots', exist_ok=True)
plt.savefig('tau_plots/log10_tau_grid.png')
plt.close()

# plot log10(gamma)
plt.figure()
sns.heatmap(loggamma, xticklabels=[f"{d:.0e}" for d in d_vals], yticklabels=[f"{m:.0e}" for m in m_vals],
            cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label':'log10(gamma [1/s])'})
plt.xlabel('d (m)'); plt.ylabel('m (kg)')
plt.title('log10(gamma) grid')
plt.tight_layout()
plt.savefig('tau_plots/log10_gamma_grid.png')
plt.close()

# save annotated CSV with human readable columns
with open('tau_plots/tau_grid_annotated.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m(kg)','d(m)','DeltaE(J)','tau(s)','gamma(1/s)','log10_tau','log10_gamma'])
    for r in rows:
        tau = float(r[3]); gamma = float(r[4])
        ltau = math.log10(tau) if np.isfinite(tau) and tau>0 else ''
        lg = math.log10(gamma) if np.isfinite(gamma) and gamma>0 else ''
        writer.writerow([r[0], r[1], r[2], r[3], r[4], ltau, lg])

print("Saved plots in ./tau_plots and annotated CSV.")
