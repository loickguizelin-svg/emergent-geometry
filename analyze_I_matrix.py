"""
analyze_I_matrix.py
Usage: python analyze_I_matrix.py
Remplace la matrice I_synthetic par ta matrice I réelle ou charge un fichier.
Produit : figures/avgI_heatmap.png, figures/Iij_reordered.png, figures/mds_embeddings.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import scipy.cluster.hierarchy as sch

# --- paramètres ---
alpha = 1.0            # paramètre pour D_ij = exp(-alpha * I_ij)
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

# --- Exemple synthétique (remplace par ton chargement) ---
# I doit être une matrice NxN symétrique, diagonale ~ 0
I = np.array([
    [0.0, 1.0, 0.2, 0.1],
    [1.0, 0.0, 0.15, 0.05],
    [0.2, 0.15, 0.0, 0.9],
    [0.1, 0.05, 0.9, 0.0]
], dtype=float)

# Si tu as un fichier .npz contenant I: charger ainsi
# data = np.load('data/I_matrix.npz'); I = data['I']

# --- sanity checks ---
assert I.shape[0] == I.shape[1], "I must be square"
if not np.allclose(I, I.T, atol=1e-8):
    print("Warning: I is not symmetric within tolerance; symmetrizing.")
    I = 0.5 * (I + I.T)
# clip small negatives
I = np.clip(I, 0.0, None)

N = I.shape[0]

# --- 1. Heatmap annotée (valeurs) ---
plt.figure(figsize=(5,4))
vmin, vmax = 0.0, np.max(I)
sns.heatmap(I, cmap='viridis', annot=True, fmt='.2f', square=True,
            vmin=vmin, vmax=vmax, cbar_kws={'label':'I_ij (bits)'})
plt.title('Mutual information matrix I_ij (t)')
plt.xlabel('Subsystem j'); plt.ylabel('Subsystem i')
plt.tight_layout()
plt.savefig(f"{fig_dir}/Iij_heatmap_annotated.png", dpi=300)
plt.close()

# --- 2. Réordonnancement par clustering hiérarchique (pour faire apparaître blocs) ---
# utiliser linkage sur la distance (1 - normalized I) ou sur I directement
# ici on construit une distance simple
# éviter division par zéro : normaliser par max
I_norm = I / (np.max(I) + 1e-12)
dist_for_clust = 1.0 - I_norm  # plus I élevé -> plus proche -> petite distance
linkage = sch.linkage(sch.distance.squareform(dist_for_clust), method='average')
idx = sch.leaves_list(linkage)
I_reordered = I[np.ix_(idx, idx)]

plt.figure(figsize=(5,4))
sns.heatmap(I_reordered, cmap='viridis', annot=True, fmt='.2f', square=True,
            vmin=vmin, vmax=vmax, cbar_kws={'label':'I_ij (bits)'})
plt.title('I_ij reordered by hierarchical clustering')
plt.xlabel('reordered j'); plt.ylabel('reordered i')
plt.tight_layout()
plt.savefig(f"{fig_dir}/Iij_reordered.png", dpi=300)
plt.close()

# --- 3. Calcul de W (normalisation par ligne) et D_eff ---
row_sums = I.sum(axis=1)
eps = 1e-12
W = np.zeros_like(I)
for i in range(N):
    s = row_sums[i]
    if s > eps:
        W[i,:] = I[i,:] / s
    else:
        W[i,:] = 0.0  # ligne sans information
# symétriser si on veut une version non directionnelle
W_sym = 0.5 * (W + W.T)

# valeurs propres (réelles attendues)
eigvals = np.linalg.eigvals(W_sym)
eigvals = np.real_if_close(eigvals)
eigvals = np.clip(eigvals, 0.0, None)  # éliminer petites négatives numériques
s1 = np.sum(eigvals)
s2 = np.sum(eigvals**2)
D_eff = (s1**2 / s2) if s2 > 0 else 0.0

print(f"D_eff = {D_eff:.4f}")
# sauvegarder W et D_eff
np.savez_compressed(f"{fig_dir}/W_and_Deff.npz", W=W, W_sym=W_sym, D_eff=D_eff, eigvals=eigvals)

# --- 4. Matrice de distances D_ij et embedding MDS ---
D = np.exp(-alpha * I)  # distances dans (0,1]
# MDS (précomputed dissimilarity)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
X = mds.fit_transform(D)  # coordonnées 2D
stress = mds.stress_
print(f"MDS stress = {stress:.6f}")

# plot embedding
plt.figure(figsize=(5,4))
plt.scatter(X[:,0], X[:,1], s=120, c='C0')
for i,(x,y) in enumerate(X):
    plt.text(x, y, f' {i}', va='center', fontsize=12)
plt.title('MDS embedding from D_ij = exp(-alpha * I_ij)')
plt.xlabel('X1'); plt.ylabel('X2')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{fig_dir}/mds_embedding.png", dpi=300)
plt.close()

# --- 5. Composite figure (heatmaps + embeddings) ---
fig, axes = plt.subplots(2,2, figsize=(8,7))
sns.heatmap(I, cmap='viridis', ax=axes[0,0], cbar_kws={'label':'I_ij (bits)'}, vmin=vmin, vmax=vmax)
axes[0,0].set_title('I_ij (t=0)')
sns.heatmap(I_reordered, cmap='viridis', ax=axes[0,1], cbar_kws={'label':'I_ij (bits)'}, vmin=vmin, vmax=vmax)
axes[0,1].set_title('I_ij (reordered)')
axes[1,0].scatter(X[:,0], X[:,1], s=120, c='C0')
for i,(x,y) in enumerate(X): axes[1,0].text(x, y, f' {i}', va='center')
axes[1,0].set_title('MDS embedding')
# show Deff and stress
axes[1,1].axis('off')
txt = f"D_eff = {D_eff:.3f}\nMDS stress = {stress:.3e}\nalpha = {alpha}"
axes[1,1].text(0.05, 0.6, txt, fontsize=12)
axes[1,1].set_title('Diagnostics')
plt.tight_layout()
plt.savefig(f"{fig_dir}/composite_I_analysis.png", dpi=300)
plt.close()

print("All figures saved in:", fig_dir)
