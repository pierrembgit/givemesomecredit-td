# Give Me Some Credit - Kaggle

## 1. Objectifs

L'objectif de cette compétition Kaggle est de construire un modèle capable de prédire sur la base d'un historique de 250 000 lignes si un empreunteur va se trouver en détresse financière dans les deux prochaines années.

Le critère d'évaluation retenu dans cette compététion est l'**Area Under the Curve (AUC)**

## 2. Structure du notebook

### 2.0 Dataviz

### 2.1 Prétraitements

Le premier notebook **S1_Pretraitements** effectue les opérations suivantes sur les fichiers source (Train set et Test set) fournis par kaggle :
 - Aligenement des colonnes sur les deux dataset
 - Remplacement des _NaN_ par des 0 dans les colonnes "NumberOfDependents" et "MonthlyIncome"
 - Remplacement des valeurs "aberrante" (>= 20) par la valeur médiane de chaque dataset pour les colonnes :
    - NumberOfTime30-59DaysPastDueNotWorse
    - NumberOfTime60-89DaysPastDueNotWorse
    - NumberOfTimes90DaysLate
 - Création d'une nouvelle feature "NumberOfTimeGlobal" qui consolide les données les trois colonnes "NumberOfTime_xxx" en appliquant les coefficients suivants :
    - 1 x NumberOfTime30-59DaysPastDueNotWorse
    - 2 x NumberOfTime60-89DaysPastDueNotWorse
    - 3 x NumberOfTimes90DaysLate
 - Suppression des trois colonnes précédemment citées
 - Création d'une nouvelle feature "IsOld" affectant la valeur 1 aux personnes dont l'age est >= 70 ans et 0 aux autres personnes.
 - Renommage des colonnes "Unnamed : 0" en "Id"
 
 Les deux dataset sont ensuite sauvegardés dans des csv distincts pour être manipulés dans les notebook suivants.

 ### 2.1 ML