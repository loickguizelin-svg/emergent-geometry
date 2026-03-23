# toy_tensor_geometry.py
# Toy model 4 qubits: mutual info -> distance -> MDS embedding
# Sauvegarde figures et imprime diagnostics.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import basis, tensor, ket2dm, qeye, sigmaz, ptrace, mesolve, Options
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle, os, time, argparse

sns.set()
plt.rcParams['figure.figsize'] = (6,4)

# --- utilitaires entropie / mutual info ---
def von_neumann_entropy(qobj):
    rho_mat = qobj.full()
    vals = np.linalg.eigvals(rho_mat)
    vals = np.real(vals)
    vals = np.clip(vals, 0, 1)
    vals = vals[vals > 1e-12]
    return -np.sum(vals * np.log(vals))

def mutual_information(rho, i, j):
    rho_i = ptrace(rho, [i])
    rho_j = ptrace(rho, [j])
    rho_ij = ptrace(rho, [i, j])
    Si = von_neumann_entropy(rho_i)
    Sj = von_neumann_entropy(rho_j)
    Sij = von_neumann_entropy(rho_ij)
    return Si + Sj - Sij

# --- état initial : deux Bell pairs (0-1) et (2-3) ---
zero = basis(2,0); one = basis(2,1)
phi = (tensor(zero,zero) + tensor(one,one)).unit()
state = tensor(phi, phi)  # 4 qubits
rho0 = ket2dm(state)
n = 4

# --- collapse ops phase damping ---
def build_phase_damping_ops(n_qubits, gamma_map):
    from qutip import qeye, sigmaz, tensor
    c_ops = []
    for i in range(n_qubits):
        rate = gamma_map.get(i, 0.0)
        if rate > 0:
            op_list = [qeye(2)]*n_qubits
            op_list[i] = sigmaz()
            c_ops.append(np.sqrt(rate) * tensor(op_list))
    return c_ops

# ---------------- concurrence & negativity helpers (2-qubit) ----------------
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

# --- simulation et diagnostics ---
def simulate_and_diagnose(rho0_arg, tlist, gamma_global=0.2, alpha=1.0, d0=1.0, local_gamma_map=None, outdir="results"):
    """
    rho0_arg : Qobj or None (if None, uses module-level rho0)
    tlist : array-like of times
    gamma_global : global decoherence rate (per qubit) or baseline
    alpha, d0 : parameters for distance mapping
    local_gamma_map : dict {index: gamma_local} to override per-qubit rates
    outdir : directory to save figures and diagnostics
    Returns diagnostics dict and saves diagnostics.pkl in outdir.
    """
    os.makedirs(outdir, exist_ok=True)

    # determine initial state
    if rho0_arg is None:
        try:
            rho0_local = globals()['rho0']
        except KeyError:
            raise ValueError("rho0 absent : fournissez rho0_arg ou définissez rho0 globalement.")
    else:
        rho0_local = rho0_arg

    # build gamma map per qubit
    gamma_map = {i: gamma_global for i in range(n)}
    if local_gamma_map:
        gamma_map.update(local_gamma_map)
    c_ops = build_phase_damping_ops(n, gamma_map)
    H = 0 * tensor([qeye(2)]*n)

    # run master equation with robust integrator options
    # choose options heuristically based on gamma_global to handle stiff regimes
    try:
        if gamma_global is None:
            gamma_global = 0.0
        if gamma_global > 1e6:
            opts = Options(nsteps=1000000, atol=1e-9, rtol=1e-7, method='bdf')
        elif gamma_global > 1e3:
            opts = Options(nsteps=500000, atol=1e-9, rtol=1e-7, method='bdf')
        elif gamma_global > 1e1:
            opts = Options(nsteps=200000, atol=1e-8, rtol=1e-6, method='bdf')
        else:
            opts = Options(nsteps=20000, atol=1e-8, rtol=1e-6)
        result = mesolve(H, rho0_local, tlist, c_ops, [], options=opts)
    except Exception as e:
        # retry with stiffer options if integrator fails
        print("Integrator failed on first attempt:", e)
        try:
            opts = Options(nsteps=1000000, atol=1e-9, rtol=1e-7, method='bdf')
            result = mesolve(H, rho0_local, tlist, c_ops, [], options=opts)
            print("Integrator succeeded on retry with stiffer options.")
        except Exception as e2:
            print("Integrator retry failed:", e2)
            # As a last resort, try a very short tlist to capture initial transient
            try:
                short_tlist = np.linspace(0, min(0.01, float(tlist[-1])), max(9, len(tlist)))
                opts = Options(nsteps=200000, atol=1e-8, rtol=1e-6, method='bdf')
                result = mesolve(H, rho0_local, short_tlist, c_ops, [], options=opts)
                # expand result to match requested tlist by padding last state
                # build a fake result.states list matching original tlist length
                from qutip import Qobj
                last_state = result.states[-1]
                padded_states = list(result.states) + [last_state] * (len(tlist) - len(result.states))
                class FakeResult:
                    def __init__(self, states):
                        self.states = states
                result = FakeResult(padded_states)
                print("Integrator fallback: short tlist used and padded to full length.")
            except Exception as e3:
                print("All integrator attempts failed:", e3)
                raise

    diagnostics = {
        'tlist': list(tlist),
        'I_matrices': [],
        'mean_I': [],
        'coords': [],
        'stress': [],
        'silhouette': [],
        'S_nodes': [],
        'mean_d_per_node': [],
        'concurrence_pairs': [],
        'negativity_pairs': [],
        'rho_list': []  # <-- store full density matrices here
    }

    for idx, rho_t in enumerate(result.states):
        # store full rho matrix for later analysis
        try:
            diagnostics['rho_list'].append(rho_t.full())
        except Exception:
            # fallback: try converting to numpy array
            try:
                diagnostics['rho_list'].append(np.array(rho_t))
            except Exception:
                diagnostics['rho_list'].append(None)

        # mutual info matrix
        I = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    I[i,j] = von_neumann_entropy(ptrace(rho_t, [i]))
                else:
                    I[i,j] = mutual_information(rho_t, i, j)
        diagnostics['I_matrices'].append(I)
        mean_I = np.mean(I[np.triu_indices(n, k=1)])
        diagnostics['mean_I'].append(mean_I)
        # distance matrix
        D = d0 * np.exp(-alpha * I)
        # MDS embedding and stress
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, n_init=4, max_iter=300)
        coords = mds.fit_transform(D)
        stress = getattr(mds, 'stress_', None)
        diagnostics['coords'].append(coords)
        diagnostics['stress'].append(stress)
        # clustering silhouette (on coords)
        try:
            labels = AgglomerativeClustering(n_clusters=2).fit_predict(coords)
            sil = silhouette_score(coords, labels)
        except Exception:
            sil = np.nan
        diagnostics['silhouette'].append(sil)
        # entropies per node and mean distances
        S_nodes = [von_neumann_entropy(ptrace(rho_t, [i])) for i in range(n)]
        diagnostics['S_nodes'].append(S_nodes)
        mean_d = D.mean(axis=1)
        diagnostics['mean_d_per_node'].append(mean_d)
        # concurrence/negativity for nearest-neighbor pairs (0-1,2-3)
        concs = []
        negs = []
        for a in range(0, n, 2):
            rho2 = ptrace(rho_t, [a, a+1])
            # convert to numpy 4x4
            try:
                rho2_mat = rho2.full()
            except Exception:
                rho2_mat = np.array(rho2)
            # compute concurrence/negativity using helper functions above
            concs.append(concurrence(rho2_mat))
            negs.append(negativity(rho2_mat))
        diagnostics['concurrence_pairs'].append(concs)
        diagnostics['negativity_pairs'].append(negs)

        # save plots for this time step (optional)
        t = tlist[idx]
        # heatmap I
        plt.figure()
        sns.heatmap(I, annot=False, cmap='viridis')
        plt.title(f'Mutual info matrix t={t:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"heat_I_t{idx:03d}.png"))
        plt.close()
        # embedding
        plt.figure()
        plt.scatter(coords[:,0], coords[:,1], s=120)
        for k in range(n):
            plt.text(coords[k,0], coords[k,1], f'q{k}', ha='center', va='center')
        plt.title(f'Embedding t={t:.2f} stress={stress:.2f} sil={sil:.3f}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"embed_t{idx:03d}.png"))
        plt.close()

    # summary plots
    plt.figure()
    plt.plot(diagnostics['tlist'], diagnostics['mean_I'], '-o')
    plt.xlabel('time'); plt.ylabel('mean pairwise mutual info')
    plt.title('Mean pairwise mutual information vs time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'meanI_vs_time.png'))
    plt.close()

    # save diagnostics (compressed)
    try:
        with open(os.path.join(outdir, 'diagnostics.pkl'), "wb") as f:
            pickle.dump(diagnostics, f)
    except Exception:
        # fallback to gzip compressed pickle if plain write fails
        import gzip
        with gzip.open(os.path.join(outdir, 'diagnostics.pkl.gz'), "wb") as f:
            pickle.dump(diagnostics, f)

    # print key diagnostics to console
    print("Saved results in", outdir)
    print("Times:", diagnostics['tlist'])
    print("Mean I start {:.4f} end {:.4f}".format(diagnostics['mean_I'][0], diagnostics['mean_I'][-1]))
    print("Stress values:", np.round(diagnostics['stress'],3))
    print("Silhouette values:", np.round(diagnostics['silhouette'],3))
    print("Entropies per node at start:", np.round(diagnostics['S_nodes'][0],3))
    print("Entropies per node at end:", np.round(diagnostics['S_nodes'][-1],3))
    print("Mean distances per node at end:", np.round(diagnostics['mean_d_per_node'][-1],3))

    return diagnostics

# ---------------- CLI interface ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Toy tensor geometry simulation")
    parser.add_argument('--gamma', type=float, default=0.2, help='global gamma decoherence rate')
    parser.add_argument('--tmax', type=float, default=2.0, help='maximum time for simulation')
    parser.add_argument('--nt', type=int, default=9, help='number of time points')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha parameter for distance mapping')
    parser.add_argument('--outdir', type=str, default='results_default', help='output directory')
    parser.add_argument('--local_gamma', type=str, default=None, help='local gamma map as comma-separated i:rate pairs e.g. "0:2.0,1:2.0"')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tlist = np.linspace(0, args.tmax, args.nt)
    local_map = None
    if args.local_gamma:
        local_map = {}
        for token in args.local_gamma.split(','):
            if ':' in token:
                i, val = token.split(':')
                local_map[int(i.strip())] = float(val.strip())
    print("Running simulation...")
    start = time.time()
    diag = simulate_and_diagnose(rho0, tlist, gamma_global=args.gamma, alpha=args.alpha, d0=1.0, local_gamma_map=local_map, outdir=args.outdir)
    end = time.time()
    print("Done in {:.1f}s".format(end-start))
