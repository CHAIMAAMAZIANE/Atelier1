Atelier 1 : Deep Learning avec PyTorch — Régression et Classification Multi-Classes

Formation : Master Sécurité IT & Big Data — FST Tanger
Module : Deep Learning
Encadrant : Pr. Lotfi ELAACHAK

1. Présentation du projet

Cet atelier a pour objectif de mettre en pratique le Deep Learning à travers deux problèmes d'apprentissage supervisé : la régression et la classification multi-classes.

Le travail est organisé en deux parties. La première partie porte sur la régression à partir de données boursières historiques. La deuxième partie sera ajoutée au même dépôt GitHub lorsqu'elle sera réalisée.

Les principaux objectifs sont :

Comprendre et préparer un jeu de données.

Réaliser une analyse exploratoire des données (EDA).

Construire et entraîner un réseau de neurones avec PyTorch.

Optimiser les hyperparamètres avec Scikit-learn et GridSearchCV.

Étudier les techniques de régularisation pour limiter le surapprentissage.

Évaluer et comparer les performances des modèles.

2. Outils utilisés

Python : traitement et analyse des données.

Pandas et NumPy : manipulation des données.

Matplotlib : visualisation des résultats.

PyTorch : construction et entraînement des réseaux de neurones.

Scikit-learn : préparation des données, recherche d'hyperparamètres et évaluation.

Google Colab / Kaggle : environnement d'exécution.

GitHub : gestion et partage du code.

3. Partie 1 : Régression sur les données boursières NYSE

3.1. Objectif

Cette partie consiste à construire un modèle de Deep Learning capable de prédire le prix de clôture de la prochaine séance boursière à partir de données historiques.

Le jeu de données NYSE contient notamment les prix d'ouverture, les prix hauts et bas, les prix de clôture, les volumes échangés et les dates.

3.2. Analyse exploratoire des données

L'analyse exploratoire permet de comprendre la structure du dataset avant l'entraînement. Elle comprend l'étude des statistiques descriptives, des valeurs manquantes, des distributions et des corrélations, ainsi que l'évolution des cours boursiers.

3.3. Préparation des données

Des variables supplémentaires sont créées à partir des prix et des volumes, notamment les rendements historiques, les distances par rapport aux moyennes mobiles et la volatilité récente.

Les données sont organisées chronologiquement afin de respecter l'ordre temporel. La standardisation est ajustée sur les données d'entraînement, puis appliquée aux données de validation et de test.

3.4. Construction et optimisation du modèle

Un réseau de neurones de type MLP (Multi-Layer Perceptron) est développé avec PyTorch. Il utilise des couches entièrement connectées, des fonctions d'activation ReLU et une sortie numérique adaptée à la régression.

La fonction de perte utilisée est la MSE (Mean Squared Error). La recherche d'hyperparamètres avec GridSearchCV permet de comparer différentes configurations, notamment le taux d'apprentissage, l'optimiseur, le nombre d'époques et l'architecture du réseau.

3.5. Régularisation et évaluation

Plusieurs techniques sont étudiées pour limiter le surapprentissage :

Dropout : désactive aléatoirement certains neurones pendant l'entraînement.

Régularisation L2 : pénalise les poids trop importants grâce à weight_decay.

Batch Normalization : normalise les activations intermédiaires du réseau.

Early Stopping : arrête l'entraînement lorsque les performances sur la validation ne s'améliorent plus.

Les modèles sont évalués à l'aide de métriques de régression telles que la MSE, le coefficient (R^2) et la RMSE. La précision directionnelle permet également d'étudier si le modèle prédit correctement le sens de variation du prix.
