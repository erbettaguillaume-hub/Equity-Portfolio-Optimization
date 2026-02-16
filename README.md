# Equity Portfolio Optimisation (Streamlit)

Application pour :
- optimiser un portefeuille long-only via Markowitz min-vol sous contrainte de rendement,
- visualiser la frontière efficiente + nuage Monte Carlo,
- backtester out-of-sample vs un benchmark et calculer des métriques de performance.

> Données de marché récupérées via yfinance (source : :contentReference[oaicite:2]{index=2}).  
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
