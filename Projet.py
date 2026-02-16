import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
MC_PORTFOLIOS = 10_000
FRONTIER_POINTS = 50
MIN_TRAIN_OBS = 60
MIN_TEST_OBS = 20


# =========================
# Fonctions utilitaires
# =========================
def parse_tickers(raw_value: str) -> tuple[list[str], list[str]]:
    tickers: list[str] = []
    duplicates: set[str] = set()

    for token in raw_value.split(","):
        ticker = token.strip().upper()
        if not ticker:
            continue
        if ticker in tickers:
            duplicates.add(ticker)
            continue
        tickers.append(ticker)

    return tickers, sorted(duplicates)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator / denominator)


def format_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def download_close_prices(tickers: list[str], start_date: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start_date,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"]
    else:
        if "Close" in data.columns:
            col_name = tickers[0] if len(tickers) == 1 else "CLOSE"
            close = data[["Close"]].rename(columns={"Close": col_name})
        else:
            close = data.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close.columns = [str(col).upper() for col in close.columns]
    return close.dropna(how="all").sort_index()


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    cov_values = cov_matrix.values
    return float(np.sqrt(weights @ cov_values @ weights))


def optimize_min_vol(
    avg_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    n_assets = len(avg_rets)
    x0 = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    mu = avg_rets.values

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {
            "type": "ineq",
            "fun": lambda w, mu=mu, target=target_return: float(np.dot(mu, w) - target),
        },
    ]

    result = minimize(
        lambda w, cov=cov_matrix: portfolio_volatility(w, cov),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        message = result.message if result.message else "Optimisation impossible."
        raise ValueError(str(message))

    return result.x, x0, bounds


def compute_performance_metrics(
    returns: pd.Series,
    bench_returns: pd.Series,
    rf: float = RISK_FREE_RATE,
) -> tuple[dict[str, str], pd.Series, pd.Series, float, float]:
    aligned = pd.concat(
        [returns.rename("portfolio"), bench_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("Pas assez de données communes pour calculer les métriques.")

    portfolio = aligned["portfolio"]
    benchmark = aligned["benchmark"]

    ann_ret = float(np.exp(portfolio.mean() * TRADING_DAYS) - 1.0)
    ann_vol = float(portfolio.std() * np.sqrt(TRADING_DAYS))

    downside_vol = float(portfolio[portfolio < 0].std() * np.sqrt(TRADING_DAYS))
    sharpe = safe_divide(ann_ret - rf, ann_vol)
    sortino = safe_divide(ann_ret - rf, downside_vol)

    cum_prices = np.exp(portfolio.cumsum())
    dd_series = cum_prices.div(cum_prices.cummax()).sub(1.0)
    max_dd = float(dd_series.min()) if not dd_series.empty else np.nan
    calmar = safe_divide(ann_ret, abs(max_dd)) if not pd.isna(max_dd) else np.nan

    bench_var = float(benchmark.var())
    beta = safe_divide(float(portfolio.cov(benchmark)), bench_var)

    bench_ann_ret = float(benchmark.mean() * TRADING_DAYS)
    alpha = np.nan
    if not pd.isna(beta):
        alpha = ann_ret - (rf + beta * (bench_ann_ret - rf))

    metrics = {
        "Rendement Annuel": format_pct(ann_ret),
        "Volatilité Annuelle": format_pct(ann_vol),
        "Ratio de Sharpe": format_num(sharpe),
        "Ratio de Sortino": format_num(sortino),
        "Ratio de Calmar": format_num(calmar),
        "Max Drawdown": format_pct(max_dd),
        "Bêta CAPM": format_num(beta),
        "Alpha CAPM (Ann.)": format_pct(alpha),
    }

    return metrics, dd_series, cum_prices, ann_ret, ann_vol


def simulate_portfolios(
    avg_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = MC_PORTFOLIOS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.random.random((n_portfolios, len(avg_rets)))
    weights /= weights.sum(axis=1, keepdims=True)

    sim_returns = weights @ avg_rets.values
    sim_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix.values, weights))
    sim_sharpe = np.divide(
        sim_returns,
        sim_vols,
        out=np.full_like(sim_returns, np.nan),
        where=sim_vols > 0,
    )

    return sim_returns, sim_vols, sim_sharpe


def compute_efficient_frontier(
    avg_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    bounds: list[tuple[float, float]],
    x0: np.ndarray,
    min_return: float,
    max_return: float,
    n_points: int = FRONTIER_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    if max_return < min_return:
        min_return, max_return = max_return, min_return

    if np.isclose(min_return, max_return):
        targets = np.array([min_return])
    else:
        targets = np.linspace(min_return, max_return, n_points)

    frontier_returns: list[float] = []
    frontier_vols: list[float] = []
    mu = avg_rets.values

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, mu=mu, target=target: float(np.dot(mu, w) - target),
            },
        ]

        result = minimize(
            lambda w, cov=cov_matrix: portfolio_volatility(w, cov),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            frontier_returns.append(float(target))
            frontier_vols.append(float(result.fun))

    return np.array(frontier_returns), np.array(frontier_vols)


def main() -> None:
    st.set_page_config(
        page_title="Portail optimisation de portefeuille & backtest",
        layout="wide",
        page_icon="📈",
    )

    st.markdown(
        """
        <style>
        .main { background-color: #fcfcfc; }
        div.stButton > button:first-child {
            background-color: #1e3a8a;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        h1, h2, h3 { color: #0f172a; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Terminal d'Optimisation")
    st.markdown("Optimisation Min-Volatilité | Frontière Efficiente | Benchmarking")


    # =========================
    # Sidebar
    # =========================
    with st.sidebar:
        st.header("Configuration")
    
        ticker_input = st.text_area(
            "1. Vos Actifs (séparés par virgule) :",
            value="AAPL, MSFT, NVDA, MC.PA, ASML.AS, BTC-USD",
            height=100,
        )
        tickers, duplicate_tickers = parse_tickers(ticker_input)
    
        st.markdown("---")
    
        bench_ticker = st.text_input("2. Benchmark (Ticker Yahoo) :", value="^GSPC").strip().upper()
    
        st.markdown("---")
    
        st.subheader("3. Périodes")
        start_date = st.date_input("Début Historique", pd.to_datetime("2020-01-01"))
        test_start_date = st.date_input("Début Backtest", pd.to_datetime("2024-01-01"))
    
        st.markdown("---")
    
        st.subheader("4. Contrainte")
        target_return_pct = st.number_input(
            "Rendement annuel minimum (%)",
            min_value=-100.0,
            max_value=1000.0,
            value=10.0,
            step=0.5,
        )
    
        run_button = st.button("Lancer l'Analyse", type="primary")
    
    
    # =========================
    # Exécution
    # =========================
    if not run_button:
        st.info("Saisissez vos paramètres et lancez l'analyse.")
        st.stop()
    
    if duplicate_tickers:
        st.warning(f"Tickers dupliqués ignorés : {', '.join(duplicate_tickers)}")
    
    if not tickers:
        st.error("Veuillez renseigner au moins un actif.")
        st.stop()
    
    if not bench_ticker:
        st.error("Veuillez renseigner un benchmark.")
        st.stop()
    
    analysis_start = pd.Timestamp(start_date)
    backtest_start = pd.Timestamp(test_start_date)
    target_return = target_return_pct / 100.0
    
    if analysis_start >= backtest_start:
        st.error("Le début du backtest doit être strictement après le début de l'historique.")
        st.stop()
    
    if target_return_pct > 100:
        st.warning("Objectif de rendement très élevé. L'optimisation peut échouer.")
    
    with st.spinner("Calcul en cours..."):
        requested_tickers = list(dict.fromkeys(tickers + [bench_ticker]))
        prices = download_close_prices(requested_tickers, analysis_start)
    
    if prices.empty:
        st.error("Aucune donnée récupérée. Vérifiez les tickers et la période.")
        st.stop()
    
    available_columns = set(prices.columns)
    valid_tickers = [ticker for ticker in tickers if ticker in available_columns]
    missing_tickers = [ticker for ticker in tickers if ticker not in available_columns]
    
    if missing_tickers:
        st.warning(f"Tickers indisponibles ignorés : {', '.join(missing_tickers)}")
    
    if not valid_tickers:
        st.error("Aucun actif valide disponible après téléchargement.")
        st.stop()
    
    if bench_ticker not in available_columns:
        st.error(f"Benchmark indisponible : {bench_ticker}")
        st.stop()
    
    asset_prices = prices[valid_tickers]
    bench_prices = prices[bench_ticker]
    
    asset_returns = np.log(asset_prices / asset_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    bench_returns = np.log(bench_prices / bench_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    
    train_rets = asset_returns.loc[asset_returns.index < backtest_start].dropna(how="any")
    test_rets = asset_returns.loc[asset_returns.index >= backtest_start].dropna(how="any")
    test_bench = bench_returns.loc[bench_returns.index >= backtest_start].dropna()
    
    common_test_index = test_rets.index.intersection(test_bench.index)
    test_rets = test_rets.loc[common_test_index]
    test_bench = test_bench.loc[common_test_index]
    
    if len(train_rets) < MIN_TRAIN_OBS:
        st.error(
            f"Pas assez de données d'entraînement ({len(train_rets)} obs). "
            f"Minimum recommandé : {MIN_TRAIN_OBS}."
        )
        st.stop()
    
    if len(test_rets) < MIN_TEST_OBS:
        st.error(
            f"Pas assez de données de backtest ({len(test_rets)} obs). "
            f"Minimum recommandé : {MIN_TEST_OBS}."
        )
        st.stop()
    
    cov_matrix = train_rets.cov() * TRADING_DAYS
    avg_rets = train_rets.mean() * TRADING_DAYS
    
    if avg_rets.isna().any() or cov_matrix.isna().any().any():
        st.error("Données insuffisantes après nettoyage. Essayez moins d'actifs ou une période plus longue.")
        st.stop()
    
    try:
        opt_w, x0, bounds = optimize_min_vol(avg_rets, cov_matrix, target_return)
    except ValueError as exc:
        st.error(
            "Optimisation impossible avec les paramètres actuels. "
            f"Détail: {exc}. Essayez de baisser le rendement minimum."
        )
        st.stop()
    
    vol_theorique = portfolio_volatility(opt_w, cov_matrix)
    ret_theorique = float(np.dot(avg_rets.values, opt_w))
    
    portfolio_test_rets = test_rets.dot(opt_w)
    
    try:
        metrics_portfolio, dd_portfolio, cum_portfolio, real_ret, real_vol = compute_performance_metrics(
            portfolio_test_rets,
            test_bench,
        )
        metrics_benchmark, dd_benchmark, cum_benchmark, _, _ = compute_performance_metrics(
            test_bench,
            test_bench,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    
    
    # =========================
    # Affichages
    # =========================
    st.header("Allocation Optimale du Capital")
    
    weight_df = pd.DataFrame(
        {
            "Actif": valid_tickers,
            "Poids (%)": opt_w * 100,
        }
    ).sort_values(by="Poids (%)", ascending=False)
    
    fig_weights = px.bar(
        weight_df,
        x="Actif",
        y="Poids (%)",
        text="Poids (%)",
        color="Poids (%)",
        color_continuous_scale="Turbo",
        template="plotly_white",
    )
    fig_weights.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1.5,
    )
    fig_weights.update_layout(
        coloraxis_showscale=True,
        height=500,
        xaxis=dict(title="Actifs", tickangle=-45),
        yaxis=dict(title="Allocation (%)"),
    )
    st.plotly_chart(fig_weights, use_container_width=True)
    
    st.header(f"Performance Réelle vs Benchmark ({bench_ticker})")
    
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Indicateur</b>",
                        "<b>Votre Portefeuille</b>",
                        f"<b>Benchmark ({bench_ticker})</b>",
                    ],
                    fill_color="#1e3a8a",
                    align="left",
                    font=dict(color="white", size=14),
                ),
                cells=dict(
                    values=[
                        list(metrics_portfolio.keys()),
                        list(metrics_portfolio.values()),
                        list(metrics_benchmark.values()),
                    ],
                    fill_color="#f8f9fa",
                    align="left",
                    font=dict(size=13),
                    height=30,
                ),
            )
        ]
    )
    fig_table.update_layout(margin=dict(l=0, r=0, b=10, t=10), height=320)
    st.plotly_chart(fig_table, use_container_width=True)
    
    st.subheader(f"Évolution de la Valeur (vs {bench_ticker})")
    
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio,
            name="Votre Portefeuille",
            line=dict(color="#00CC96", width=3),
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=cum_benchmark.index,
            y=cum_benchmark,
            name=f"Benchmark ({bench_ticker})",
            line=dict(color="gray", dash="dot"),
        )
    )
    fig_line.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        height=500,
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("Analyse du Drawdown")
    
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd_portfolio.index,
            y=dd_portfolio * 100,
            fill="tozeroy",
            name="Portefeuille",
            line=dict(color="#e24141"),
        )
    )
    fig_dd.add_trace(
        go.Scatter(
            x=dd_benchmark.index,
            y=dd_benchmark * 100,
            name="Benchmark",
            line=dict(color="gray"),
        )
    )
    fig_dd.update_layout(
        yaxis_title="Baisse (%)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.subheader("Nuage Monte Carlo & Frontière Efficiente")
    
    sim_returns, sim_vols, sim_sharpe = simulate_portfolios(avg_rets, cov_matrix)
    frontier_returns, frontier_vols = compute_efficient_frontier(
        avg_rets,
        cov_matrix,
        bounds,
        x0,
        min_return=ret_theorique,
        max_return=float(avg_rets.max()),
    )
    
    fig_mc = px.scatter(
        x=sim_vols,
        y=sim_returns,
        color=sim_sharpe,
        color_continuous_scale="Jet",
        labels={"x": "Volatilité", "y": "Rendement"},
    )
    
    if len(frontier_vols) > 0:
        fig_mc.add_trace(
            go.Scatter(
                x=frontier_vols,
                y=frontier_returns,
                mode="lines",
                line=dict(color="black", width=3),
                name="Frontière Efficiente",
            )
        )
    
    fig_mc.add_trace(
        go.Scatter(
            x=[vol_theorique],
            y=[ret_theorique],
            mode="markers",
            marker=dict(color="white", size=15, symbol="star", line=dict(width=2, color="black")),
            name="Théorie (Passé)",
        )
    )
    fig_mc.add_trace(
        go.Scatter(
            x=[real_vol],
            y=[real_ret],
            mode="markers",
            marker=dict(color="#FFD900", size=18, symbol="diamond", line=dict(width=2, color="black")),
            name="Réalité (Backtest)",
        )
    )
    fig_mc.update_layout(
        height=600,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        template="plotly_white",
    )
    st.plotly_chart(fig_mc, use_container_width=True)


if __name__ == "__main__":
    main()
