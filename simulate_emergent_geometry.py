#!/usr/bin/env python3
"""
simulate_emergent_geometry.py

Toy simulation for emergent geometry with local internal dimensions:
- 4 visible qubits
- N_anc ancillas per visible (default N_anc = 3)
- Local phase-damping on all qubits
- Sweep gamma (25 log-spaced values)
- Compute mutual information between visibles, D_eff, MDS embedding, stress, t1/2
Outputs: PNG figures and results_summary.csv
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.manifold import MDS
from qutip import *
from qutip import entropy_vn, ptrace


# -------------------------
# User parameters
# -------------------------
N_VISIBLE = 4
N_ANC = 1            # ancillas per visible (you chose 3)
REDUCED_MODE = True # set True to reduce memory (N_ANC -> 2, fewer gammas)
ALPHA = 1.0
TMAX = 5.0
N_T = 50

# Gamma sweep
if REDUCED_MODE:
    GAMMAS = np.logspace(-6, 1, 9)
    N_ANC = max(1, N_ANC - 1)
else:
    GAMMAS = np.logspace(-6, 1, 25)

# Derived
N_ANC_TOTAL = N_VISIBLE * N_ANC
N_TOTAL = N_VISIBLE + N_ANC_TOTAL
TIMES = np.linspace(0, TMAX, N_T)

OUT_DIR = "New_output exp_protege"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# Helper utilities
# -------------------------
def tensor_op(single_op, target, n_total):
    """Return operator acting as single_op on qubit target in n_total-qubit space."""
    ops = [qeye(2) for _ in range(n_total)]
    ops[target] = single_op
    return tensor(ops)

def ancilla_index(visible_idx, anc_idx):
    """Global index of ancilla anc_idx (0..N_ANC-1) for visible visible_idx (0..N_VISIBLE-1)."""
    return N_VISIBLE + visible_idx * N_ANC + anc_idx

# -------------------------
# Build initial state
# -------------------------
# For each block (visible + its ancillas) create GHZ-like (|0...0> + |1...1>)/sqrt(2)
block_states = []
for i in range(N_VISIBLE):
    # local block dimension = 1 visible + N_ANC ancillas
    zero_state = tensor([basis(2,0) for _ in range(1 + N_ANC)])
    one_state  = tensor([basis(2,1) for _ in range(1 + N_ANC)])
    ghz = (zero_state + one_state).unit()
    block_states.append(ghz)

# Full tensor product of blocks (ordering: block0 (v0,a0..), block1 (v1,a1..), ...)
psi_blocks = tensor(block_states)
rho0 = ket2dm(psi_blocks)
rho_init = ket2dm(psi_blocks)

# Now entangle visible qubits into Bell pairs (0-1) and (2-3) on visible subspace
# We'll apply Hadamard on visible 0 then CNOT(0->1), and similarly for 2->3.
from qutip import Qobj
import numpy as np

H = 0 * tensor([qeye(2) for _ in range(N_TOTAL)])   # Hamiltonien nul de la bonne dimension

CNOT_4 = Qobj(np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]]))

from qutip import tensor, qeye

def apply_two_qubit_unitary(rho, U, idx1, idx2):
    """
    Applique l'unitaire 4x4 U sur les qubits idx1 et idx2 (indices globaux) de rho.
    Méthode : construire l'unitaire complet en tensorisant identités et U aux bonnes positions.
    """
    # indices à placer dans l'ordre croissant pour construire la liste d'opérateurs
    i1, i2 = int(idx1), int(idx2)
    if i1 == i2:
        raise ValueError("idx1 and idx2 must be different")
    # on construit la liste d'opérateurs en parcourant les qubits
    ops = []
    for k in range(N_TOTAL):
        if k == i1:
            ops.append(None)   # placeholder A
        elif k == i2:
            ops.append(None)   # placeholder B
        else:
            ops.append(qeye(2))
    # on remplace les deux placeholders par l'unitaire 2-qubit U via une factorisation:
    # construire la liste de facteurs en deux étapes : d'abord remplacer par I pour tensor,
    # puis appliquer une permutation via permutation_operator (option A) ou construire U_full autrement.
    # Ici on utilise la permutation trick (nécessite permutation_operator importé).
    perm = list(range(N_TOTAL))
    perm.remove(i1); perm.remove(i2)
    perm = [i1, i2] + perm
    P = permutation_operator([2]*N_TOTAL, perm)
    Pdag = P.dag()
    U_full = Pdag * tensor(U, qeye(2**(N_TOTAL-2))) * P
    return U_full * rho * U_full.dag()


# -------------------------
# Collapse operators (phase damping on each qubit)
# -------------------------
sigmaz_q = sigmaz()
#def build_collapse_ops(gamma):
 #   Ls = []
  #  sqrt_g = math.sqrt(gamma)
  #  for k in range(N_TOTAL):
  #      Ls.append(sqrt_g * tensor_op(sigmaz_q, k, N_TOTAL))
   # return Ls

def build_collapse_ops(gamma):
    """
    On applique la décohérence (bruit) UNIQUEMENT sur les qubits visibles.
    Les ancillas sont dans une 'dimension privée' et restent protégées.
    """
    ops = []
    # N_TOTAL est normalement déjà défini globalement dans ton script
    for i in range(N_VISIBLE):
        # Création d'un opérateur qui ne cible que le qubit visible 'i'
        list_op = [qeye(2)] * N_TOTAL
        list_op[i] = sigmaz()
        # On ajoute l'opérateur à la liste des "destructeurs"
        ops.append(math.sqrt(gamma) * tensor(list_op))
    return ops



# -------------------------
# Diagnostics helpers
# -------------------------
def pairwise_mutual_information(full_rho):
    """Compute symmetric matrix I_ij for visible qubits only (size N_VISIBLE x N_VISIBLE)."""
    I = np.zeros((N_VISIBLE, N_VISIBLE))
    for i in range(N_VISIBLE):
        for j in range(i+1, N_VISIBLE):
            # trace out all but i and j from full_rho
            rho_ij = ptrace(full_rho, [i, j])
            # compute entropies (von Neumann, base 2)
            S_ij = entropy_vn(rho_ij, base=2)
            # single-qubit reduced states from rho_ij
            rho_i = ptrace(rho_ij, 0)
            rho_j = ptrace(rho_ij, 1)
            S_i = entropy_vn(rho_i, base=2)
            S_j = entropy_vn(rho_j, base=2)
            Iij = max(0.0, S_i + S_j - S_ij)
            I[i, j] = Iij
            I[j, i] = Iij
    return I

def effective_dimension_from_W(I):
    """Compute W row-normalized and participation ratio D_eff from eigenvalues."""
    W = np.zeros_like(I)
    for i in range(I.shape[0]):
        s = np.sum(I[i, :])
        if s > 0:
            W[i, :] = I[i, :] / s
        else:
            W[i, :] = 0.0
    # eigenvalues
    lambdas = np.real(np.linalg.eigvals(W))
    if np.sum(lambdas**2) > 0:
        D_eff = (np.sum(lambdas)**2) / np.sum(lambdas**2)
    else:
        D_eff = 0.0
    return D_eff, W

def mds_and_stress(Dmat):
    """Return 2D coords and classical MDS stress for given dissimilarity matrix Dmat."""
    # ensure diagonal zero
    np.fill_diagonal(Dmat, 0.0)
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, normalized_stress='auto')
        coords = mds.fit_transform(Dmat)
        dij = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        stress = math.sqrt(np.sum((dij - Dmat)**2) / np.sum(Dmat**2)) if np.sum(Dmat**2) > 0 else 0.0
        return coords, stress
    except Exception:
        return None, np.nan

# -------------------------
# Main sweep
# -------------------------
results = []
timeseries = {}

for gamma in GAMMAS:
    print(f"Running gamma = {gamma:.3e}")
    Ls = build_collapse_ops(gamma)
    # Solve master equation (H = 0)
    try:
        out = mesolve(H=H, rho0=rho_init, tlist=TIMES, c_ops=Ls, options=Options(nsteps=10000))

    except Exception as e:
        print("mesolve failed:", e)
        break
    states = out.states

    avgI_t = np.zeros(len(TIMES))
    D_eff_t = np.zeros(len(TIMES))
    stress_t = np.zeros(len(TIMES))
    I_time_series = np.zeros((len(TIMES), N_VISIBLE, N_VISIBLE))

    for ti, r in enumerate(states):
        I_mat = pairwise_mutual_information(r)
        I_time_series[ti] = I_mat
        # average over unique pairs
        pairs = []
        for i in range(N_VISIBLE):
            for j in range(i+1, N_VISIBLE):
                pairs.append(I_mat[i, j])
        avgI = np.mean(pairs) if len(pairs) > 0 else 0.0
        avgI_t[ti] = avgI
        D_eff, W = effective_dimension_from_W(I_mat)
        D_eff_t[ti] = D_eff
        Dmat = np.exp(-ALPHA * I_mat)
        coords, stress = mds_and_stress(Dmat)
        stress_t[ti] = stress if stress is not None else np.nan

    # half-life t1/2
    I0 = avgI_t[0]
    half = I0 / 2.0
    t_half = np.nan
    if I0 > 0 and np.any(avgI_t <= half):
        # find first crossing
        idx = np.where(avgI_t <= half)[0][0]
        if idx == 0:
            t_half = TIMES[0]
        else:
            t_half = np.interp(half, [avgI_t[idx-1], avgI_t[idx]], [TIMES[idx-1], TIMES[idx]])
    results.append({
        "gamma": float(gamma),
        "t_half": float(t_half) if not math.isnan(t_half) else np.nan,
        "I0": float(I0),
        "Iend": float(avgI_t[-1]),
        "D_eff0": float(D_eff_t[0]),
        "D_eff_end": float(D_eff_t[-1]),
        "stress0": float(stress_t[0]),
        "stress_end": float(stress_t[-1])
    })
    timeseries[gamma] = {
        "times": TIMES,
        "avgI": avgI_t,
        "D_eff": D_eff_t,
        "stress": stress_t,
        "I_times": I_time_series
    }

    # Save per-gamma figures
    plt.figure()
    plt.plot(TIMES, avgI_t, '-o', markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Average mutual information (visibles)")
    plt.title(f"avgI vs t, gamma={gamma:.1e}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"avgI_gamma_{gamma:.1e}.png"))
    plt.close()

    plt.figure()
    plt.plot(TIMES, D_eff_t, '-o', markersize=3)
    plt.xlabel("Time")
    plt.ylabel("D_eff")
    plt.title(f"D_eff vs t, gamma={gamma:.1e}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"Deff_gamma_{gamma:.1e}.png"))
    plt.close()

    # heatmaps at t=0 and t=end
    plt.figure()
    plt.imshow(I_time_series[0], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"I_ij at t=0, gamma={gamma:.1e}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"I0_gamma_{gamma:.1e}.png"))
    plt.close()

    plt.figure()
    plt.imshow(I_time_series[-1], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"I_ij at t=end, gamma={gamma:.1e}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"Iend_gamma_{gamma:.1e}.png"))
    plt.close()

# Summary CSV
import csv
csv_path = os.path.join(OUT_DIR, "results_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gamma","t_half","I0","Iend","D_eff0","D_eff_end","stress0","stress_end"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Save timeseries dict as numpy file
np.save(os.path.join(OUT_DIR, "timeseries_dict.npy"), timeseries)

# Global summary plot: t_half vs gamma
gammas = np.array([r["gamma"] for r in results])
t_halfs = np.array([r["t_half"] if not math.isnan(r["t_half"]) else np.nan for r in results])
plt.figure()
plt.loglog(gammas, np.where(np.isnan(t_halfs), 1e-12, t_halfs), marker='o')
plt.xlabel("gamma")
plt.ylabel("t_half")
plt.title("Half-life vs gamma")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "t_half_vs_gamma.png"))
plt.close()

print("Simulation finished. Outputs in:", OUT_DIR)
