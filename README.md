# Optimisation de portefeuilles (Streamlit)

## Présentation de l'application
    - Optimiser un portefeuille long-only via Markowitz min-vol sous contrainte de rendement
    - Visualiser la frontière efficiente + nuage Monte Carlo
    - Backtester out-of-sample vs un benchmark et calculer des métriques de performance.

> Données de marché récupérées via yfinance
> **Disclaimer** : projet éducatif, pas un conseil en investissement.

## Fonctionnalités
    - Optimisation **min-volatilité** (SLSQP, contraintes : somme des poids = 1, poids ∈ [0,1], rendement cible)
    - Frontière efficiente + simulation Monte Carlo de portefeuilles
    - Backtest buy & hold out-of-sample
    - Indicateurs : Sharpe, Sortino, Calmar, Max Drawdown, bêta CAPM, alpha de Jensen

## Installation des librairies nécessiares
pip install -r requirements.txt

## Lancer l'application
Après avoir exécuter le code, écrire dans le terminal la commande suivante : streamlit run projet.py

## Paramètres dans la barre latérale
    - Les actifs : liste séparée par virgules (ex: AAPL, MSFT, NVDA)
    - Benchmark : ticker Yahoo (ex: ^GSPC)
    - Début Historique : période d’entraînement
    - Début Backtest : période out-of-sample
    - Rendement annuel minimum : contrainte pour l’optimisation

## Hypothèses et limites
    - Stratégie long-only et buy & hold sur la période de test (pas de rebalancing).
    - Les rendements sont calculés en log-returns.
    - La qualité des résultats dépend de la liquidité, de la disponibilité des données et de la cohérence des tickers.
