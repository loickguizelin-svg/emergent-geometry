# mera_toy.py
# Toy MERA (N=8) -> compute mutual info, distances, MDS embedding, decoherence tests.
import numpy as np
import scipy.linalg as la
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, time

sns.set()
plt.rcParams['figure.figsize'] = (6,4)

# ---------------- utilities ----------------
def kronN(mats):
    """Kronecker product of list of matrices"""
    out = np.array([1.0])
    for M in mats:
        out = np.kron(out, M)
    return out

def random_unitary(dim):
    X = (np.random.randn(dim,dim) + 1j*np.random.randn(dim,dim))/np.sqrt(2)
    Q, R = la.qr(X)
    # fix phase
    D = np.diag(np.diag(R)/np.abs(np.diag(R)))
    return Q @ la.inv(D)

def random_isometry(in_dim, out_dim):
    # isometry W: in_dim -> out_dim with out_dim <= in_dim (we'll use out_dim < in_dim)
    X = (np.random.randn(in_dim, out_dim) + 1j*np.random.randn(in_dim, out_dim))/np.sqrt(2)
    Q, R = la.qr(X)
    return Q[:, :out_dim]

def partial_trace(rho, keep, dims):
    # rho: full density matrix (Ndim x Ndim)
    # keep: list of subsystem indices to keep (0-based)
    # dims: list of dims per subsystem (e.g., [2,2,2,...])
    # returns reduced density matrix on subsystems 'keep'
    total = int(np.prod(dims))
    assert rho.shape == (total, total)
    # reshape to tensor
    resh = rho.reshape(dims + dims)
    # axes: (i0,i1,...,iN, j0,j1,...,jN)
    N = len(dims)
    keep_set = set(keep)
    trace_axes = [i for i in range(N) if i not in keep_set]
    # move axes so that traced ones are contiguous
    perm = [i for i in range(N) if i in keep] + trace_axes + [N + i for i in range(N) if i in keep] + [N + i for i in range(N) if i not in keep]
    resh_perm = np.transpose(resh, perm)
    k = len(keep)
    dim_keep = int(np.prod([dims[i] for i in keep]))
    dim_trace = int(np.prod([dims[i] for i in trace_axes])) if trace_axes else 1
    resh2 = resh_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    # trace over middle dims
    red = np.trace(resh2, axis1=1, axis2=3)
    return red

def von_neumann_entropy(rho):
    vals = np.linalg.eigvals(rho)
    vals = np.real(vals)
    vals = np.clip(vals, 1e-12, None)
    return -np.sum(vals * np.log(vals))

def mutual_information(rho_full, i, j, dims):
    rho_i = partial_trace(rho_full, [i], dims)
    rho_j = partial_trace(rho_full, [j], dims)
    rho_ij = partial_trace(rho_full, [i,j], dims)
    Si = von_neumann_entropy(rho_i)
    Sj = von_neumann_entropy(rho_j)
    Sij = von_neumann_entropy(rho_ij)
    return Si + Sj - Sij

# ---------------- build toy MERA ----------------
def build_toy_mera_state(N=8, phys_dim=2, seed=None):
    # N must be power of 2 for this simple MERA
    if seed is not None:
        np.random.seed(seed)
    # We'll build a 2-layer MERA for N=8:
    # layer 0: disentanglers U (4 of them acting on pairs), isometries W mapping pairs->single effective (2->1)
    # layer 1: disentanglers on 4 sites -> isometries -> top state
    # For simplicity we implement a unitary circuit that mimics MERA structure.
    # Start from product state |0...0>, apply layers of disentanglers (2-qubit unitaries) and isometries (2->1 via projection)
    # We'll implement isometry as a 2->1 map by embedding into 2-qubit unitary then projecting.
    # Simpler: apply 2-qubit unitaries in a MERA pattern to create entanglement.
    # This is a heuristic toy MERA.
    # Initialize product |0>^N
    zero = np.array([1.0, 0.0], dtype=complex)
    psi = kronN([zero]*N)  # state vector length 2^N
    # define 2-qubit random unitaries
    def apply_two_qubit_unitary(state, U, site, N):
        # U acts on qubits (site, site+1)
        # build full operator
        ops = []
        for k in range(N):
            if k == site:
                ops.append(None)  # placeholder for two-qubit block
            elif k == site+1:
                ops.append(None)
            else:
                ops.append(np.eye(2))
        # construct kron with insertion
        left = kronN(ops[:site]) if site>0 else np.array([1.0])
        right = kronN(ops[site+2:]) if site+2 < N else np.array([1.0])
        full = np.kron(np.kron(left, U), right)
        return full @ state
    # MERA-like pattern: layer of disentanglers on even bonds, then odd bonds
    # create random 2-qubit unitaries
    U_even = [random_unitary(4) for _ in range(N//2)]
    U_odd = [random_unitary(4) for _ in range((N//2)-1)]
    # apply even bonds (0-1,2-3,...)
    for b in range(0, N, 2):
        psi = apply_two_qubit_unitary(psi, U_even[b//2], b, N)
    # apply odd bonds (1-2,3-4,...)
    for b in range(1, N-1, 2):
        psi = apply_two_qubit_unitary(psi, U_odd[(b-1)//2], b, N)
    # second layer: repeat pattern to mimic coarse-graining
    U_even2 = [random_unitary(4) for _ in range(N//2)]
    for b in range(0, N, 2):
        psi = apply_two_qubit_unitary(psi, U_even2[b//2], b, N)
    # normalize
    psi = psi / la.norm(psi)
    return psi

# ---------------- decoherence channels (Kraus) ----------------
def phase_damping_kraus(p):
    # single-qubit phase damping Kraus operators
    K0 = np.array([[1,0],[0,np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0,0],[0,np.sqrt(p)]], dtype=complex)
    return [K0, K1]

def apply_local_kraus(rho, kraus_map, N):
    # kraus_map: dict {site: [K0,K1,...]} for sites with noise; others identity
    # dims all 2
    full_ops = []
    for i in range(N):
        if i in kraus_map:
            full_ops.append(kraus_map[i])
        else:
            full_ops.append([np.eye(2)])
    # apply channel: rho' = sum_{k0,...,kN} (K0⊗...⊗KN) rho (K0⊗...⊗KN)^\dagger
    # naive implementation: iterate over product of indices (exponential), but N small (<=12)
    from itertools import product
    Ks_lists = full_ops
    new_rho = np.zeros_like(rho)
    for indices in product(*[range(len(Ks)) for Ks in Ks_lists]):
        ops = [Ks_lists[i][indices[i]] for i in range(N)]
        Kfull = kronN(ops)
        new_rho += Kfull @ rho @ Kfull.conj().T
    return new_rho

# ---------------- concurrence & negativity for 2-qubit ----------------
def concurrence(rho2):
    rho = np.array(rho2, dtype=complex)
    # Wootters
    sigma_y = np.array([[0,-1j],[1j,0]])
    Y = np.kron(sigma_y, sigma_y)
    rho_tilde = Y @ rho.conj() @ Y
    R = rho @ rho_tilde
    eigs = np.real(np.linalg.eigvals(R))
    vals = np.sqrt(np.clip(np.sort(eigs)[::-1], 0, None))
    C = max(0.0, vals[0] - vals[1] - vals[2] - vals[3])
    return C

def negativity(rho2):
    rho = np.array(rho2, dtype=complex)
    rho_resh = rho.reshape(2,2,2,2)
    rho_pt = np.transpose(rho_resh, (0,3,2,1)).reshape(4,4)
    eigs = np.linalg.eigvals(rho_pt)
    neg = np.sum(np.abs(eigs[eigs < 0]))
    return np.real(neg)

# ---------------- main experiment ----------------
def run_experiment(N=8, tlist=None, gamma_global=0.2, alpha=1.0, outdir="mera_results", seed=42):
    os.makedirs(outdir, exist_ok=True)
    if tlist is None:
        tlist = np.linspace(0,2,9)
    # build state
    psi = build_toy_mera_state(N=N, seed=seed)
    rho0 = np.outer(psi, psi.conj())
    dims = [2]*N
    diagnostics = {'tlist': tlist, 'I_matrices': [], 'mean_I': [], 'coords': [], 'stress': [], 'silhouette': [], 'S_nodes': [], 'mean_d_per_node': [], 'concurrence_pairs': [], 'negativity_pairs': []}
    for idx, t in enumerate(tlist):
        # map gamma from t (simple model): p = 1 - exp(-gamma_global * t)
        p = 1 - np.exp(-gamma_global * t)
        # build kraus map: same p on all sites
        kraus_map = {i: phase_damping_kraus(p) for i in range(N)}
        rho_t = apply_local_kraus(rho0, kraus_map, N)
        # compute mutual info matrix for single-site subsystems
        I = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    S = von_neumann_entropy(partial_trace(rho_t, [i], dims))
                    I[i,j] = S
                else:
                    I[i,j] = mutual_information(rho_t, i, j, dims)
        diagnostics['I_matrices'].append(I)
        meanI = np.mean(I[np.triu_indices(N, k=1)])
        diagnostics['mean_I'].append(meanI)
        # distances
        D = np.exp(-alpha * I)
        # MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, n_init=4, max_iter=300)
        coords = mds.fit_transform(D)
        stress = getattr(mds, 'stress_', None)
        diagnostics['coords'].append(coords)
        diagnostics['stress'].append(stress)
        # silhouette on coords
        try:
            labels = AgglomerativeClustering(n_clusters=2).fit_predict(coords)
            sil = silhouette_score(coords, labels)
        except Exception:
            sil = np.nan
        diagnostics['silhouette'].append(sil)
        # entropies per node
        S_nodes = [von_neumann_entropy(partial_trace(rho_t, [i], dims)) for i in range(N)]
        diagnostics['S_nodes'].append(S_nodes)
        mean_d = D.mean(axis=1)
        diagnostics['mean_d_per_node'].append(mean_d)
        # concurrence/negativity for nearest-neighbor pairs (0-1,2-3,...)
        concs = []
        negs = []
        for a in range(0, N, 2):
            rho2 = partial_trace(rho_t, [a, a+1], dims)
            concs.append(concurrence(rho2))
            negs.append(negativity(rho2))
        diagnostics['concurrence_pairs'].append(concs)
        diagnostics['negativity_pairs'].append(negs)
        # save heatmap and embedding
        plt.figure(); sns.heatmap(I, annot=False, cmap='viridis'); plt.title(f'Mutual info t={t:.2f}'); plt.tight_layout(); plt.savefig(f"{outdir}/heat_I_{idx:03d}.png"); plt.close()
        plt.figure(); plt.scatter(coords[:,0], coords[:,1], s=80); 
        for k in range(N): plt.text(coords[k,0], coords[k,1], f'{k}', ha='center', va='center')
        plt.title(f'Embedding t={t:.2f} stress={stress:.3f} sil={sil:.3f}'); plt.tight_layout(); plt.savefig(f"{outdir}/embed_{idx:03d}.png"); plt.close()
    # summary plot mean I
    plt.figure(); plt.plot(tlist, diagnostics['mean_I'], '-o'); plt.xlabel('time'); plt.ylabel('mean pairwise mutual info'); plt.grid(True); plt.tight_layout(); plt.savefig(f"{outdir}/meanI_vs_time.png"); plt.close()
    # save diagnostics
    with open(f"{outdir}/diagnostics.pkl", 'wb') as f:
        pickle.dump(diagnostics, f)
    print("Saved results in", outdir)
    return diagnostics

# ---------------- run if main ----------------
if __name__ == "__main__":
    start = time.time()
    diag = run_experiment(N=8, tlist=np.linspace(0,2,9), gamma_global=0.6, alpha=1.0, outdir="mera_results", seed=123)
    end = time.time()
    print("Done in {:.1f}s".format(end-start))
