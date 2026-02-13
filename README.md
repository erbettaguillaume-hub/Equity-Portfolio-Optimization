# Equity-Portfolio-Optimisation

Voici une application Streamlit regroupant une optimisation de portefeuilles ainsi qu'un backtesting "out-of-sample" de la stratégie optimale

L'objectif de ce projet est de mettre en pratique l'optimisation d'un portefeuille d'actions selon le modèle min-variance de Markowitz.
On obtient à partir des données d'entraînements :
  - Un diagramme en bâton des poids optimaux pour chacune des actions
  - Une simulation MonteCarlo de portefeuilles permettant de mettre en lumière le portfeuille optimal et la frontière efficiente
    
Dans un second temps sur les données de test:
  - Déterminer le Béta de ce portefeuille optimal
  - Déterminer son "alpha de Jensen" à partir du CAPM
  - Calculer des indicateurs de performance (ratio de Sharpe, ratio de Sortino et ratio de Calmar)

Ensuite, à partir des données de test, l'application contient également :
  - une comparaions de l'évolution du rendement du portefeuille par rapport à un benchmark
  - une analyse des drawdowns du portefeuille comparé à ceux du benchmark
