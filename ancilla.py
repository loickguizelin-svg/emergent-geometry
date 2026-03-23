# -*- coding: utf-8 -*-
"""
Simulation de géométrie émergente avec dimensions internes privées (ancillas).
Pour N≤3 : utilisation de mesolve (exact).
Pour N≥4 : utilisation de mcsolve (Monte Carlo) avec average_states.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import qutip as qt

# ============================
# Paramètres modifiables
# ============================
N = 4                     # Nombre de particules (chaque particule = 1 visible + 1 ancilla)
                         # Total = 2N qubits
t_max = 5.0               # Temps maximal (réduit pour mcsolve)
dt = 0.1                  # Pas de temps
gamma = 0.5               # Taux de décohérence (phase damping)
alpha = 1.0               # Facteur d'échelle pour distance D = exp(-alpha * I)

# Choix de l'état initial
# 0 = état |0> pour tous (pas d'intrication)
# 1 = intrication maximale visible-ancilla pour chaque particule (|Φ+>)
etat_initial_choice = 1

# Pour mcsolve : nombre de trajectoires (plus = meilleure précision, mais plus long)
ntraj = 100 if N >= 4 else 1  # 1 pour mesolve (pas utilisé)

# ============================
# Construction des opérateurs
# ============================
dim_total = 2**(2*N)       # dimension de l'espace total

# Opérateurs de Pauli pour un qubit
si = qt.qeye(2)
sz = qt.sigmaz()

# Opérateurs de collapse pour le phase damping sur chaque qubit
c_ops = []
for i in range(2*N):
    op_list = [si] * (2*N)
    op_list[i] = sz
    c_ops.append(np.sqrt(gamma) * qt.tensor(op_list))

# ============================
# État initial
# ============================
if etat_initial_choice == 0:
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(2*N)])
elif etat_initial_choice == 1:
    # Intrication maximale entre chaque visible et son ancilla
    # Les paires sont (0,1), (2,3), (4,5), ...
    composantes = []
    for i in range(N):
        bell = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) +
                qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
        composantes.append(bell)
    psi0 = qt.tensor(composantes)
else:
    psi0 = qt.tensor([qt.basis(2,0) for _ in range(2*N)])

psi0 = psi0.unit()

# ============================
# Simulation temporelle
# ============================
times = np.arange(0, t_max, dt)
print(f"Simulation de l'évolution pour N={N}...")

# Hamiltonien nul
H0 = qt.Qobj(np.zeros((dim_total, dim_total)), dims=[psi0.dims[0], psi0.dims[0]])

if N <= 3:
    # Utiliser mesolve (exact)
    result = qt.mesolve(H0, psi0, times, c_ops, [])
    states = result.states
else:
    # Utiliser mcsolve avec average_states pour obtenir directement la matrice densité moyenne
    options = {'average_states': True, 'progress_bar': True}
    result = qt.mcsolve(H0, psi0, times, c_ops, [], ntraj=ntraj, options=options)
    states = result.states  # Liste de Qobj (matrices densité moyennées)

print("Calcul des informations mutuelles et de la géométrie...")
I_ij_history = []
stress_history = []

# Indices des qubits visibles (pairs)
vis_indices = list(range(0, 2*N, 2))

for i, rho in enumerate(states):
    # Vérifier que rho est un Qobj (il devrait l'être)
    if not isinstance(rho, qt.Qobj):
        print(f"Attention: l'état à l'instant {i} n'est pas un Qobj mais un {type(rho)}. On ignore.")
        continue

    # Réduction sur les visibles
    rho_vis = rho.ptrace(vis_indices)

    # Calcul des entropies individuelles
    S = np.zeros(N)
    for j in range(N):
        rho_j = rho_vis.ptrace([j])
        S[j] = qt.entropy_vn(rho_j)

    # Information mutuelle
    I_ij = np.zeros((N, N))
    for j in range(N):
        for k in range(j+1, N):
            rho_jk = rho_vis.ptrace([j, k])
            S_jk = qt.entropy_vn(rho_jk)
            I_ij[j, k] = S[j] + S[k] - S_jk
            I_ij[k, j] = I_ij[j, k]

    I_ij_history.append(I_ij)

    # Matrice de distances
    D = np.exp(-alpha * I_ij)
    np.fill_diagonal(D, 0)

    # MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, normalized_stress='auto')
    try:
        coords = mds.fit_transform(D)
        stress = mds.stress_
    except:
        stress = np.nan
    stress_history.append(stress)

print("Simulation terminée.")

# ============================
# Visualisation
# ============================
I_mean = [np.mean(I[np.triu_indices(N, k=1)]) for I in I_ij_history]

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(times[:len(I_mean)], I_mean)  # times peut être plus long si des états ont été ignorés
plt.xlabel('Temps')
plt.ylabel('Information mutuelle moyenne (visibles)')
plt.title('Décroissance des corrélations visibles')

plt.subplot(1,2,2)
plt.plot(times[:len(stress_history)], stress_history)
plt.xlabel('Temps')
plt.ylabel('Stress MDS')
plt.title('Qualité de la projection géométrique')
plt.tight_layout()
plt.show()

# Visualisation de la géométrie à différents instants
indices = [0, len(I_ij_history)//4, len(I_ij_history)//2, -1]
plt.figure(figsize=(12,3))
for idx, t_idx in enumerate(indices):
    plt.subplot(1,4,idx+1)
    I = I_ij_history[t_idx]
    D = np.exp(-alpha * I)
    np.fill_diagonal(D, 0)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, normalized_stress='auto')
    coords = mds.fit_transform(D)
    plt.scatter(coords[:,0], coords[:,1], s=100)
    for j in range(N):
        plt.annotate(str(j), (coords[j,0], coords[j,1]), fontsize=12)
    plt.title(f't = {times[t_idx]:.1f}')
    plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()