# Projet Frozen Lake — Value Iteration & Policy Iteration

Ce projet applique deux algorithmes de planification (Value Iteration et Policy Iteration) sur l’environnement FrozenLake-v1 de Gymnasium.
L’objectif est de trouver la politique optimale permettant à un agent d’atteindre la case but sans tomber dans un trou, tout en comparant les performances des deux méthodes.

## 1. Objectifs :

Comprendre et implémenter les deux algorithmes.

Étudier la convergence et la stabilité des valeurs d’état.

Analyser l’effet du facteur de discount (γ), du seuil de convergence (θ) et du paramètre is_slippery.

## 2. Méthodes :

Value Iteration : mise à jour directe des valeurs d’état jusqu’à convergence.

Policy Iteration : alternance entre évaluation et amélioration de politique.

## 3. Résultats :

Les deux méthodes trouvent une politique optimale.

Policy Iteration converge en moins d’itérations, mais chaque étape est plus coûteuse.

Value Iteration est plus simple et plus stable, mais plus lente.

Quand is_slippery=True, la politique est plus prudente et le taux de réussite diminue.

## 4. Conclusion :

Policy Iteration est plus efficace pour les petits environnements.

Value Iteration reste plus intuitive et robuste.

Les performances dépendent fortement de la stochasticité et de la définition des récompenses.
