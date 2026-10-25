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



Partie 2 — Deep Learning avec PyTorch : Classification multi-classes des pannes industrielles
1. Vue d'ensemble
Cette partie consiste à construire un réseau de neurones capable de prédire le type de panne d'une machine industrielle à partir de mesures de fonctionnement : températures, vitesse de rotation, couple et usure de l'outil.

La classification comporte six classes : No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure et Random Failures.

Le principal défi est le déséquilibre des classes : plus de 96 % des observations correspondent à No Failure. Le projet compare plusieurs configurations de MLP et utilise des métriques adaptées aux classes minoritaires, notamment le rappel et le F1 macro.
2. Données et prétraitement
Le dataset initial contient 10 000 lignes et 10 colonnes. Les vérifications effectuées n'ont révélé aucune valeur manquante ni aucun doublon.

Les étapes de préparation sont les suivantes :
- Suppression de UDI et Product ID, qui sont des identifiants.
- Suppression de Target afin d'éviter une fuite de la cible : le modèle prédit Failure Type.
- Suppression de 9 observations contradictoires où Target = 1 mais Failure Type = No Failure.
- Conservation des 18 observations Random Failures, même si Target = 0, pour garder cette classe de Failure Type.
- Encodage de Type selon l'ordre L, M, H et encodage de la cible.
- Création de variables dérivées : différence entre les températures du procédé et de l'air, puissance estimée (couple × vitesse de rotation) et interaction entre l'usure et le couple.
- Détection des valeurs extrêmes par la méthode IQR. Les valeurs extrêmes observées pour la vitesse de rotation et le couple sont conservées, car elles peuvent représenter des conditions réelles de fonctionnement.
- Standardisation avec StandardScaler, ajusté uniquement sur les données d'entraînement.
Après nettoyage, 9 991 observations sont conservées.
3. Analyse exploratoire (EDA)
L'analyse exploratoire étudie la distribution des classes, les boxplots des variables selon le type de panne, les corrélations entre variables, le taux de panne par qualité de produit et la relation entre couple et vitesse de rotation.

Quelques observations relevées :
- Heat Dissipation Failure : vitesse de rotation moyenne plus faible, écart de température moyen plus faible et température de l'air plus élevée.
- Overstrain Failure : couple et usure de l'outil élevés.
- Power Failure : vitesse de rotation moyenne élevée et couple relativement élevé.
- Tool Wear Failure : usure moyenne de l'outil élevée.
- Random Failures : statistiques moyennes proches de celles des observations sans panne, ce qui rend cette classe difficile à distinguer avec les variables disponibles.
- Qualité du produit : le taux de panne observé est plus élevé pour la qualité L (4,05 %) que pour M (2,71 %) et H (2,40 %).
4. Découpage des données et SMOTE
Les données sont séparées de manière stratifiée en ensembles d'entraînement, de validation et de test. Le test conserve la distribution initiale des classes.

SMOTE est appliqué uniquement aux données d'entraînement afin de créer des exemples synthétiques pour les classes minoritaires. Après équilibrage, chaque classe d'entraînement compte 6 557 observations, soit 39 342 observations au total.

Effectifs avant SMOTE :
Ensemble	No Failure	Heat	Power	Overstrain	Tool Wear	Random
Train réel	6 557	76	64	53	31	12
Validation	1 157	14	12	9	5	2
Test	1 929	22	19	16	9	4


L'application de SMOTE après la séparation permet de préserver un test représentatif de la distribution réelle. Les classes Tool Wear Failure et Random Failures restent toutefois très peu représentées dans les ensembles de validation et de test ; leurs métriques sont donc sensibles à quelques erreurs de prédiction.
5. Architecture du réseau
Le modèle est un perceptron multicouche (MLP) implémenté avec PyTorch. Il comprend des couches entièrement connectées (Linear), des activations ReLU et une couche de sortie correspondant aux six classes.

L'entraînement utilise CrossEntropyLoss et des mini-batchs. Les configurations étudiées permettent de faire varier l'architecture et d'introduire des mécanismes de régularisation.
6. Recherche d'hyperparamètres
La recherche par grille explore les paramètres suivants :
Hyperparamètre	Valeurs testées
Learning rate	0.001, 0.01
Optimiseur	Adam, RMSprop
Nombre d'époques	20, 40
Architecture	(32), (64, 32), (128, 64, 32)


Le critère de sélection est le F1 macro, qui donne le même poids à chaque classe, contrairement à l'accuracy qui peut masquer les faibles performances sur les pannes rares.

La meilleure configuration relevée est :
Paramètre	Meilleure valeur
Learning rate	0.001
Optimiseur	Adam
Époques	40
Architecture	(128, 64, 32)
F1 macro en validation croisée	0.3326


Les meilleures configurations observées utilisent majoritairement 40 époques et un learning rate de 0.001. Le score de validation croisée reste à interpréter avec prudence, car certaines classes sont représentées par très peu d'exemples dans chaque pli.
7. Évaluation du modèle de base
Les matrices de confusion permettent de comparer les prédictions aux classes réelles, sur l'entraînement réel sans SMOTE et sur le test.
Matrice de confusion — entraînement réel
Le modèle classe correctement la majorité des observations No Failure (6 438 sur 6 557). Il reconnaît également les observations des classes minoritaires présentes dans l'entraînement. Cependant, certaines observations normales sont classées comme Tool Wear Failure ou Random Failures.
Matrice de confusion — test
Sur le test, les résultats corrects visibles sur la matrice sont :
Classe réelle	Prédictions correctes	Effectif
No Failure	1 869	1 929
Heat Dissipation Failure	19	22
Power Failure	16	19
Overstrain Failure	15	16
Tool Wear Failure	2	9
Random Failures	0	4


Le modèle reconnaît plusieurs types de pannes, mais ses résultats sont nettement plus faibles pour Tool Wear Failure et Random Failures. Cette différence illustre l'importance de consulter la matrice de confusion et le rappel par classe, plutôt que de se limiter à l'accuracy globale.
8. Régularisation et choix du modèle final
Plusieurs techniques sont comparées séparément et en combinaison :
- Dropout (0.3) pour réduire la dépendance du réseau à certains neurones.
- L2 / weight decay (1e-3) pour pénaliser les poids trop élevés.
- Batch Normalization pour normaliser les activations intermédiaires.
- Early stopping avec une patience de 8 époques, basé sur le F1 macro de validation.
- Une configuration combinant plusieurs techniques.
Chaque configuration est évaluée avec trois seeds afin de mesurer la variabilité des résultats. Le meilleur modèle est sélectionné à partir de la validation, et non du test.
Résultats du meilleur modèle : early stopping
Métrique	Train réel	Test
Accuracy	0.9825	0.9610
Sensibilité / Recall macro	0.9970	0.6391
Précision macro	Non renseignée	0.5122
F1 macro	0.7868	0.5614


L'accuracy du test est de 96,10 %, mais un modèle qui prédirait toujours No Failure obtiendrait environ 96,50 % sur ce jeu de test. L'accuracy seule ne suffit donc pas à juger la qualité du modèle.

Le meilleur modèle détecte plusieurs pannes : le rappel indiqué est de 0,864 pour Heat Dissipation Failure, de 0,842 pour Power Failure et de 0,938 pour Overstrain Failure. En revanche, le rappel est faible pour Tool Wear Failure (0,222) et nul pour Random Failures (0 sur 4).

Ces résultats montrent un écart de généralisation important sur les classes rares. Le modèle apprend les exemples minoritaires d'entraînement, mais peine à les reconnaître sur de nouvelles observations. Le tableau complet de comparaison des six configurations doit être consulté pour détailler l'effet moyen de chaque technique de régularisation.
9. Limites
- Les classes minoritaires disposent de très peu d'observations réelles, notamment Random Failures et Tool Wear Failure.
- Les exemples créés par SMOTE sont synthétiques et ne remplacent pas de nouvelles mesures de pannes réelles.
- Les métriques par classe sont instables lorsque le support du test est très faible.
- Les pannes aléatoires sont difficiles à distinguer à partir des variables disponibles.
- Les résultats dépendent du découpage des données et de l'initialisation du réseau.
Pour aller plus loin, il serait utile de collecter davantage d'exemples de pannes, de répéter l'évaluation avec plusieurs découpages stratifiés et d'étudier le compromis entre détection des pannes et fausses alertes.
10. Technologies utilisées
- Python
- Pandas et NumPy
- Matplotlib et Seaborn
- Scikit-learn (StandardScaler, métriques, GridSearchCV)
- PyTorch
- Imbalanced-learn (SMOTE)
- Jupyter Notebook / Google Colab
Conclusion
Cette partie met en œuvre un pipeline de classification multi-classes comprenant le nettoyage, l'analyse exploratoire, la création de variables, la standardisation, l'équilibrage SMOTE, l'entraînement d'un MLP, la recherche d'hyperparamètres et la comparaison de techniques de régularisation.

Les résultats montrent que le modèle détecte plusieurs catégories de pannes, mais que les performances restent limitées pour les classes les moins représentées. L'analyse du F1 macro, du rappel par classe et des matrices de confusion est donc essentielle pour évaluer sa capacité réelle à identifier les défaillances.



