# Emergent Geometry — Working Simulations & Research Notes

Ce dépôt rassemble les scripts Python et les éléments utilisés pour explorer une idée simple :
**et si la géométrie de l’espace-temps pouvait émerger des corrélations entre systèmes quantiques ?**

Il ne s’agit pas d’un projet académique, mais d’un **carnet de recherche ouvert**, construit pas à pas,
avec des modèles jouets, des simulations numériques et des visualisations.

L’objectif : tester une intuition, la formaliser, et la rendre reproductible.

---

## 📘 Contenu du dépôt

- `src/`  
  Scripts Python utilisés pour :
  - générer des états quantiques (visibles + degrés privés),
  - appliquer la décohérence (Lindblad),
  - calculer les corrélations (mutual information),
  - reconstruire une géométrie émergente (MDS embedding),
  - mesurer la dimension effective.

- `results/`  
  Figures, matrices, embeddings et données générées lors des tests.

- `docs/`  
  - *Working paper* (note de recherche)  
  - Résumé et explications conceptuelles

---

## 🧠 Idée générale

Dans ce modèle :
- chaque particule possède des **degrés visibles** (ce qui construit la géométrie),
- et des **degrés privés** (réservoirs d’information internes).

La géométrie n’est pas supposée :  
elle **émerge** des corrélations entre les parties visibles.

La décohérence, elle, agit comme une force qui :
- efface les corrélations,
- réduit la dimension effective,
- et peut provoquer un **effondrement géométrique**.

---

## 🧪 Pipeline de simulation

1. Génération de l’état global  
2. Trace partielle → états visibles  
3. Entropies & mutual information  
4. Normalisation (overlap matrix)  
5. Dimension effective  
6. Conversion en distances  
7. Embedding géométrique (MDS)  
8. Analyse du stress / cohérence géométrique

Chaque étape est implémentée dans les scripts Python du dossier `src/`.

---

## 🎯 Objectif du projet

Ce dépôt n’a pas vocation à “prouver” quoi que ce soit.  
C’est un **laboratoire personnel**, un espace d’exploration où je teste :

- des intuitions,
- des modèles,
- des visualisations,
- des idées issues de la physique de l’information.

Toute contribution, critique ou discussion est la bienvenue.

---

## 📄 Licence

Projet ouvert, libre d’utilisation et de modification.  
Usage académique ou personnel autorisé.

---

## ✉️ Contact

Pour échanger : via LinkedIn ou issues GitHub.
