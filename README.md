J’ai développé un terminal d’optimisation de portefeuille & backtesting en Python (Streamlit), pensé comme un mini “portfolio workstation” : on renseigne un univers d’actifs, on fixe une contrainte de rendement, et l’app construit une allocation optimale, puis teste sa stratégie out-of-sample face à un benchmark.

🎯 Objectif du projet

L’idée est simple : séparer clairement la théorie (in-sample) de la réalité (out-of-sample).

1. In-sample (période d’entraînement)
    - Estimer les paramètres de marché : rendements moyens & matrice de covariance (annualisés).
    - Résoudre un problème de Markowitz “Minimum Volatility” sous contraintes.

2. Out-of-sample (période de backtest)
    - Appliquer l’allocation obtenue sur une période de test indépendante.
    - Comparer la performance et le risque à un benchmark (ex : S&P 500).

3. Illustrer l'effet de diversification du portefeuille à l'aide des indices de performance.

---

🧠 Ce que fait concrètement l’app

1) Interface “Terminal” (Streamlit)

Depuis la sidebar, l’utilisateur définit :

   - Actifs (tickers Yahoo Finance, actions/indices/crypto possible)
   - Benchmark (ticker Yahoo)
   - Période historique (début de l’échantillon)
   - Début du backtest (split train/test)
   - Contrainte** : rendement annuel minimum (en %)

---

2) Data & preprocessing

Téléchargement des prix ajustés via yfinance
Construction des log-returns journaliers
Annualisation standard : 252 jours de trading

 Split temporel strict :

    - Train : dates < début backtest
    - Test : dates ≥ début backtest

---

3) Optimisation du portefeuille (Markowitz Min-Vol sous contrainte)

Sur la période train, l’app calcule :

    - Rendement moyen annualisé
    - Covariance annualisée

Puis elle résout :

    Objectif : minimiser la volatilité 
    Contraintes :
        - Somme des poids = 1
        - Pas de position short
        - Rendement annuel minimum

Solveur : SLSQP 

Output : poids optimaux + “point théorique” (rendement/vol) estimé in-sample.

---

4) Backtesting out-of-sample & comparaison au benchmark

Sur la période test, l’app applique les poids optimaux (allocation fixe, type buy-and-hold sur returns) et calcule :

Métriques risque/performance :

   - Rendement annualisé
   - Volatilité annualisée
   - Sharpe (avec (rf) constant, paramétré à 2% annuel)
   - Sortino
   - Max Drawdown
   - Calmar

Style / CAPM :

Estimation alpha/beta par régression OLS
Alpha affiché annualisé

📊 Visualisations :

Bar chart des poids
Table KPI Portefeuille vs Benchmark
Courbe de performance cumulée (portefeuille vs benchmark)
Courbe de drawdown comparative

---

5) Frontière efficiente + Monte Carlo

Pour contextualiser l’allocation optimale :

Simulation de 10 000 portefeuilles
Scatter Volatilité vs Rendement 
Calcul de la frontière efficiente 

Et surtout : l’app affiche côte à côte

    ⭐ le point “Théorie (Passé)” (in-sample)
    ♦️ le point “Réalité (Backtest)” (out-of-sample)

C’est une manière très visuelle de montrer l’écart entre paramètres estimés et performance réalisée.

---

🔍 Hypothèses & limites (assumées)

    Long-only, sans levier
    Pas de coûts de transaction, pas de slippage, pas de frais
    Pas de rebalancing dynamique (poids constants sur la période test)
