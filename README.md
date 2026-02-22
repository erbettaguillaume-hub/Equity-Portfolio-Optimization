# 📈 Portfolio Optimisation & Backtesting Terminal (Python / Streamlit)

J’ai développé un **terminal d’optimisation de portefeuille & backtesting** en **Python (Streamlit)**, pensé comme un mini *portfolio workstation* : l’utilisateur renseigne un univers d’actifs, fixe une **contrainte de rendement**, et l’app **construit une allocation optimale** puis **teste la stratégie out-of-sample** face à un benchmark.

---

## 🎯 Objectif du projet

L’idée est de **séparer clairement la théorie (in-sample)** de la **réalité (out-of-sample)**, tout en **illustrant l’effet de diversification** à travers des **indices de performance**.

### In-sample (période d’entraînement)
- Estimer les paramètres de marché : **rendements moyens** & **matrice de covariance** (annualisés).
- Résoudre un problème de **Markowitz “Minimum Volatility”** sous contraintes.

### Out-of-sample (période de backtest)
- Appliquer l’allocation obtenue sur une période de test indépendante.
- Comparer **performance** et **risque** à un **benchmark** (ex : S&P 500).

---

## 🧠 Ce que fait concrètement l’app

### 1) Interface “Terminal” (Streamlit)
Depuis la sidebar, l’utilisateur définit :
- **Actifs** : tickers Yahoo Finance (actions / indices / crypto)
- **Benchmark** : ticker Yahoo Finance
- **Période historique** : début de l’échantillon
- **Début du backtest** : date de split *train/test*
- **Contrainte** : **rendement annuel minimum** (en %)

---

### 2) Data & preprocessing
- Téléchargement des **prix ajustés** via `yfinance`
- Construction des **log-returns journaliers**
- **Annualisation** standard : **252 jours de trading**
- Split temporel strict :
  - **Train** : dates `<` début backtest  
  - **Test** : dates `≥` début backtest

---

### 3) Optimisation (Markowitz Min-Vol sous contrainte)
Sur la période **train**, l’app calcule :
- **Rendement moyen annualisé**
- **Covariance annualisée**

Puis elle résout :
- **Objectif** : minimiser la **volatilité**
- **Contraintes** :
  - \(\sum_i w_i = 1\) (fully invested)
  - \(0 \le w_i \le 1\) (pas de short / long-only)
  - \(\mu^\top w \ge R_{\min}\) (rendement annuel minimum)

**Solveur** : `SLSQP` (scipy)

✅ **Output** : poids optimaux + **“point théorique”** *(rendement/volatilité)* estimé **in-sample**.

---

### 4) Backtesting out-of-sample & comparaison au benchmark
Sur la période **test**, l’app applique les poids optimaux (*allocation fixe, buy-and-hold sur returns*) et calcule :

#### Métriques risque / performance
- **Rendement annualisé**
- **Volatilité annualisée**
- **Sharpe** (avec \(rf\) constant, paramétré à **2% annuel**)
- **Sortino**
- **Max Drawdown**
- **Calmar**

#### Style / CAPM
- Estimation **alpha / beta** via **régression OLS**
- **Alpha affiché annualisé**

---

## 📊 Visualisations
- **Bar chart** des poids (allocation optimale)
- **Table KPI** Portefeuille vs Benchmark
- **Courbe de performance cumulée** (portefeuille vs benchmark)
- **Courbe de drawdown** comparative

---

## 🧩 Frontière efficiente + Monte Carlo
Pour contextualiser l’allocation optimale :
- Simulation de **10 000 portefeuilles** aléatoires (long-only)
- Scatter **Volatilité vs Rendement**
- Calcul de la **frontière efficiente**

Et surtout : l’app affiche côte à côte :
- ⭐ **Point “Théorie (Passé)”** *(in-sample)*
- ♦️ **Point “Réalité (Backtest)”** *(out-of-sample)*

➡️ Une manière visuelle de montrer l’écart entre **performance estimée** et **performance réalisée**, et d’illustrer **l’effet diversification**.

---

## 🔍 Hypothèses & limites (assumées)
- **Long-only**, sans levier
- **Pas de coûts de transaction**, pas de slippage, pas de frais
- **Pas de rebalancing dynamique** (poids constants sur la période test)

---
