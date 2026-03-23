# run_gamma_sweep.py
"""
Wrapper pour lancer automatiquement des runs pour une liste de gamma,
collecter diagnostics, calculer t1/2 et produire un CSV/plots.
Usage: python run_gamma_sweep.py
Assume: tau_grid.csv présent et toy_tensor_geometry.py (ou mera_toy.py) dans le même dossier.
"""
import csv, os, sys, math, subprocess, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (6,4)

# ---------- paramètres ----------
TAU_GRID = 'tau_grid.csv'        # fichier généré précédemment
OUTDIR = 'gamma_sweep_results'
SIM_SCRIPT_NAMES = ['toy_tensor_geometry.py', 'mera_toy.py']  # ordre d'essai pour import/subprocess
SIM_FUNC_NAME = 'simulate_and_diagnose'  # nom de la fonction si import possible
TIMEOUT_PER_RUN = 1200  # secondes max par run (ajuste si besoin)
MAX_TEST = 12  # nombre max de gamma à tester (échantillonnage)

os.makedirs(OUTDIR, exist_ok=True)

# ---------- utilitaires ----------
def read_tau_grid(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # permissif sur les noms de colonnes : on cherche 5 colonnes numériques
        for r in reader:
            if len(r) < 5:
                continue
            try:
                m = float(r[0]); d = float(r[1]); dE = float(r[2]); tau = float(r[3]); gamma = float(r[4])
            except:
                try:
                    vals = [c.strip() for c in r if c.strip()!='']
                    if len(vals) < 5:
                        continue
                    m = float(vals[0]); d = float(vals[1]); dE = float(vals[2]); tau = float(vals[3]); gamma = float(vals[4])
                except:
                    continue
            rows.append({'m':m, 'd':d, 'dE':dE, 'tau':tau, 'gamma':gamma})
    return rows

def compute_t_half(tlist, meanI):
    # t1/2: time where meanI crosses half of initial meanI (linear interp)
    t = np.array(tlist)
    I = np.array(meanI)
    if len(I) == 0:
        return np.nan
    I0 = I[0]
    half = I0 / 2.0
    if np.all(I > half):
        return np.nan
    for i in range(1, len(I)):
        if I[i] <= half:
            return float(np.interp(half, [I[i-1], I[i]], [t[i-1], t[i]]))
    return np.nan

# ---------- essayer d'importer simulate_and_diagnose ----------
simulate_func = None
sim_module = None
sim_module_name = None
for name in SIM_SCRIPT_NAMES:
    if os.path.exists(name):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("sim_module", name)
            sim_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sim_module)
            if hasattr(sim_module, SIM_FUNC_NAME):
                simulate_func = getattr(sim_module, SIM_FUNC_NAME)
                sim_module_name = name
                print(f"Importé {SIM_FUNC_NAME} depuis {name}")
                break
        except Exception as e:
            print(f"Import depuis {name} échoué: {e}")
            simulate_func = None
            sim_module_name = None

# ---------- lecture de la grille et sélection des gamma à tester ----------
rows = read_tau_grid(TAU_GRID)
if len(rows) == 0:
    raise FileNotFoundError(f"Aucun couple trouvé dans {TAU_GRID}. Génère le fichier avec grid_tau.py d'abord.")

# filtrer couples valides (tau fini) ; on ignore gamma <= 0
candidates = [r for r in rows if np.isfinite(r['tau']) and r['tau']>0 and np.isfinite(r['gamma'])]
if len(candidates) == 0:
    raise ValueError("Aucun gamma valide trouvé dans tau_grid.csv")

unique_gammas = sorted({float(r['gamma']) for r in candidates})
# échantillonnage si trop nombreux
if len(unique_gammas) > MAX_TEST:
    idxs = np.linspace(0, len(unique_gammas)-1, MAX_TEST).astype(int)
    test_gammas = [unique_gammas[i] for i in idxs]
else:
    test_gammas = unique_gammas

print("Gammas testés (extrait):", test_gammas)

# ---------- boucle de runs ----------
summary = []
for gamma in test_gammas:
    print(f"\n=== Run gamma = {gamma:.3e} ===")
    run_outdir = os.path.join(OUTDIR, f"gamma_{gamma:.3e}")
    os.makedirs(run_outdir, exist_ok=True)
    start = time.time()
    diag = None

    # ignore gamma non positif explicitement
    if gamma <= 0.0:
        print(f"Gamma = {gamma} non positif détecté — on l'ignore.")
        summary.append({'gamma':gamma, 't_half':np.nan, 'stress_mean':np.nan, 'sil_mean':np.nan, 'conc_mean':np.nan, 'neg_mean':np.nan, 'elapsed_s':0.0})
        continue

    # heuristique tmax / nt :
    # - pour gamma très petit -> grande fenêtre (mais limitée)
    # - pour gamma très grand -> petite fenêtre mais résolution fine
    if gamma < 1e-6:
        tmax = min(1e5, max(2.0, 10.0 / max(gamma, 1e-18)))
        Nt = 41
    elif gamma < 1e-2:
        tmax = min(1e5, max(2.0, 10.0 / gamma))
        Nt = 41
    elif gamma > 1e4:
        tmax = min(0.01, 2.0)
        Nt = 201
    elif gamma > 1e2:
        tmax = min(0.1, 2.0)
        Nt = 201
    else:
        tmax = 2.0
        Nt = 9
    # safety caps
    if tmax > 1e5:
        tmax = 1e5
    tlist = np.linspace(0, tmax, Nt)

    if simulate_func is not None and sim_module is not None:
        # essayer de récupérer rho0 depuis le module importé
        rho0_arg = None
        if hasattr(sim_module, 'rho0'):
            rho0_arg = getattr(sim_module, 'rho0')
        # appeler la fonction en essayant plusieurs signatures possibles
        try:
            # signature la plus complète
            try:
                diag = simulate_func(rho0_arg, tlist, gamma, alpha=1.0, d0=1.0, local_gamma_map=None, outdir=run_outdir)
            except TypeError:
                # signature minimale
                try:
                    diag = simulate_func(rho0_arg, tlist, gamma)
                except TypeError:
                    # fallback sans rho0
                    diag = simulate_func(None, tlist, gamma)
        except Exception as e:
            print("Erreur lors de l'appel direct de simulate_and_diagnose:", e)
            diag = None

    if diag is None:
        # fallback : lancer le script en subprocess en lui passant --gamma et --tmax si supporté
        script_to_call = None
        for name in SIM_SCRIPT_NAMES:
            if os.path.exists(name):
                script_to_call = name
                break
        if script_to_call is None:
            raise FileNotFoundError("Aucun script simulateur trouvé pour exécuter en subprocess.")
        cmd = [sys.executable, script_to_call, '--gamma', str(gamma), '--tmax', str(tmax), '--nt', str(Nt), '--outdir', run_outdir]
        print("Lancement subprocess:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_PER_RUN)
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)
            # tenter de charger diagnostics sauvegardés par le script (diagnostics.pkl dans outdir ou results_default)
            possible_files = [os.path.join(run_outdir, 'diagnostics.pkl'), os.path.join('.', 'results_default', 'diagnostics.pkl'), 'diagnostics.pkl']
            found = None
            for pf in possible_files:
                if os.path.exists(pf):
                    found = pf; break
            if found:
                with open(found, 'rb') as f:
                    diag = pickle.load(f)
                    print("Chargé diagnostics depuis", found)
            else:
                print("Aucun diagnostics.pkl trouvé après exécution subprocess.")
                diag = None
        except subprocess.TimeoutExpired:
            print("Run timeout pour gamma", gamma)
            diag = None
        except Exception as e:
            print("Erreur subprocess:", e)
            diag = None

    elapsed = time.time() - start
    # si diag disponible, extraire métriques
    if diag is not None:
        tlist_used = np.array(diag['tlist'])
        meanI = np.array(diag['mean_I'])
        t_half = compute_t_half(tlist_used, meanI)
        stress = np.array(diag.get('stress', [np.nan]*len(tlist_used)))
        stress_mean = np.nanmean(stress)
        sil = np.array(diag.get('silhouette', [np.nan]*len(tlist_used)))
        sil_mean = np.nanmean(sil)
        # concurrence/negativity si présents
        concs = np.array(diag.get('concurrence_pairs')) if 'concurrence_pairs' in diag else None
        negs = np.array(diag.get('negativity_pairs')) if 'negativity_pairs' in diag else None
        conc_mean = np.nan if concs is None else np.nanmean(concs)
        neg_mean = np.nan if negs is None else np.nanmean(negs)
        # sauvegarde plots simples
        plt.figure(); plt.plot(tlist_used, meanI, '-o'); plt.xlabel('time'); plt.ylabel('meanI'); plt.title(f'gamma={gamma:.3e}'); plt.grid(True); plt.savefig(os.path.join(run_outdir,'meanI_vs_time.png')); plt.close()
        # heatmap first/last
        import seaborn as sns
        sns.set()
        I0 = diag['I_matrices'][0]; Iend = diag['I_matrices'][-1]
        plt.figure(); sns.heatmap(I0, cmap='viridis'); plt.title('I t0'); plt.savefig(os.path.join(run_outdir,'heat_I_t0.png')); plt.close()
        plt.figure(); sns.heatmap(Iend, cmap='viridis'); plt.title('I tend'); plt.savefig(os.path.join(run_outdir,'heat_I_tend.png')); plt.close()
        # save diag pickle copy
        with open(os.path.join(run_outdir,'diag_copy.pkl'),'wb') as f:
            pickle.dump(diag, f)
        summary.append({'gamma':gamma, 't_half':t_half, 'stress_mean':stress_mean, 'sil_mean':sil_mean, 'conc_mean':conc_mean, 'neg_mean':neg_mean, 'elapsed_s':elapsed})
        print(f"Done gamma {gamma:.3e}  t1/2={t_half} s  stress_mean={stress_mean:.3f}  sil_mean={sil_mean:.3f}")
    else:
        summary.append({'gamma':gamma, 't_half':np.nan, 'stress_mean':np.nan, 'sil_mean':np.nan, 'conc_mean':np.nan, 'neg_mean':np.nan, 'elapsed_s':elapsed})
        print("Aucun diagnostic récupéré pour gamma", gamma)

# ---------- sauvegarde résumé ----------
csv_out = os.path.join(OUTDIR, 'gamma_sweep_summary.csv')
with open(csv_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gamma','t_half_s','stress_mean','silhouette_mean','concurrence_mean','negativity_mean','elapsed_s'])
    for s in summary:
        writer.writerow([s['gamma'], s['t_half'], s['stress_mean'], s['sil_mean'], s['conc_mean'], s['neg_mean'], s['elapsed_s']])
print("Résumé sauvegardé dans", csv_out)

# ---------- plot t1/2 vs gamma ----------
gam = [s['gamma'] for s in summary]
t1 = [s['t_half'] if s['t_half'] is not None else np.nan for s in summary]
plt.figure(); plt.loglog(gam, np.abs(t1), '-o'); plt.xlabel('gamma (1/s)'); plt.ylabel('t1/2 (s)'); plt.grid(True, which='both'); plt.title('t1/2 vs gamma'); plt.savefig(os.path.join(OUTDIR,'t1half_vs_gamma.png')); plt.close()
print("Plot t1/2 vs gamma sauvegardé.")
