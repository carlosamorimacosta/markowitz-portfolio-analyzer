import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from scipy.optimize import minimize

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Markowitz | Portfolio Analytics"
)

server = app.server

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #10283F;
    --green: #28A879;
    --background: #F3F5F7;
    --border: #E2E8EE;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: var(--background);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-weight: 700;
}

h1, h2, h3 {
    color: var(--navy) !important;
}

.stButton > button {
    background: var(--navy);
    color: white;
    border-radius: 7px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1D4964;
    color: white;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# Lista inicial de tickers líquidos/conhecidos. Não é uma fotografia oficial e dinâmica
# da carteira teórica do Ibovespa; o usuário pode editar, complementar ou importar CSV.
IBOV_STARTER = [
    "ABEV3", "ALOS3", "ASAI3", "B3SA3", "BBAS3", "BBDC3", "BBDC4",
    "BBSE3", "BPAC11", "BRAP4", "BRFS3", "BRKM5", "CCRO3", "CMIG4",
    "CMIN3", "COGN3", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CYRE3",
    "ELET3", "ELET6", "EMBR3", "ENEV3", "ENGI11", "EQTL3", "GGBR4",
    "GOAU4", "HAPV3", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11",
    "LREN3", "MGLU3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PETR3",
    "PETR4", "PETZ3", "PRIO3", "RADL3", "RAIL3", "RAIZ4", "RENT3",
    "SANB11", "SBSP3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3",
    "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVA3", "VIVT3", "WEGE3"
]

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end, frequency):
    symbols = [t.strip().upper() + ".SA" for t in tickers]
    interval = "1d"
    raw = yf.download(
        symbols, start=start, end=end + timedelta(days=1), interval=interval,
        auto_adjust=True, progress=False, group_by="column", threads=True
    )
    if raw.empty:
        raise ValueError("O provedor não retornou preços. Confira os tickers e o período.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise ValueError("Não foi possível localizar a série de preços ajustados.")
    else:
        col = "Close" if "Close" in raw.columns else None
        if col is None:
            raise ValueError("Não foi possível localizar a série de preços ajustados.")
        prices = raw[[col]].copy()
        prices.columns = [symbols[0]]
    prices.columns = [str(c).replace(".SA", "") for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frequency == "Mensal":
        prices = prices.resample("ME").last()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(axis=1, how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def download_selic(start, end, frequency):
    # SGS 4390: Selic acumulada no mês (% a.m.); SGS 11: Selic diária (% a.d.).
    code = 4390 if frequency == "Mensal" else 11
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    params = {
        "formato": "json",
        "dataInicial": pd.Timestamp(start).strftime("%d/%m/%Y"),
        "dataFinal": pd.Timestamp(end).strftime("%d/%m/%Y"),
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError("O Banco Central não retornou observações da Selic para o período.")
    frame = pd.DataFrame(data)
    frame["data"] = pd.to_datetime(frame["data"], dayfirst=True)
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce") / 100
    series = frame.set_index("data")["valor"].sort_index()
    return series

def annualization_factor(frequency):
    return 12 if frequency == "Mensal" else 252

def portfolio_stats(weights, mu, cov):
    ret = float(np.dot(weights, mu))
    variance = float(weights @ cov @ weights)
    return ret, np.sqrt(max(variance, 0.0))

def optimize_mvp(mu, cov, max_weight=1.0):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Falha na otimização da MVP: {result.message}")
    return result.x

def optimize_max_sharpe(mu, cov, rf, max_weight=1.0):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    def objective(w):
        r, vol = portfolio_stats(w, mu, cov)
        return -((r - rf) / vol) if vol > 1e-12 else 1e6
    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 3000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Falha na otimização do máximo Sharpe: {result.message}")
    return result.x

def efficient_frontier(mu, cov, n_points=70, max_weight=1.0):
    w_mvp = optimize_mvp(mu, cov, max_weight)
    r_mvp, _ = portfolio_stats(w_mvp, mu, cov)
    targets = np.linspace(r_mvp, float(np.max(mu)), n_points)
    rows = []
    n = len(mu)
    bounds = [(0.0, max_weight)] * n
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t},
        ]
        result = minimize(lambda w: w @ cov @ w, w_mvp.copy(),
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 1500, "ftol": 1e-10})
        if result.success:
            ret, vol = portfolio_stats(result.x, mu, cov)
            rows.append({"Retorno": ret, "Volatilidade": vol, "Pesos": result.x})
    if not rows:
        raise RuntimeError("Não foi possível construir a fronteira eficiente.")
    return pd.DataFrame(rows)

def pct(x):
    return f"{x:.2%}"

st.title("📈 Markowitz Portfolio Analyzer")
st.caption("Otimização de carteiras de ações brasileiras • Markowitz mean-variance")

with st.sidebar:
    st.header("Configurações")
    universe_mode = st.radio("Universo de ativos", ["Ibovespa + tickers personalizados", "Somente tickers personalizados"])
    default_tickers = ", ".join(IBOV_STARTER[:20])
    if universe_mode.startswith("Ibovespa"):
        st.caption("Lista inicial editável de ações associadas ao universo do Ibovespa. Não representa composição oficial atualizada automaticamente.")
        tickers_text = st.text_area("Tickers (separados por vírgula)", value=default_tickers, height=125)
    else:
        tickers_text = st.text_area("Tickers (separados por vírgula)", value="ITUB4, WEGE3, PETR4, VALE3", height=100)
    uploaded = st.file_uploader("Ou importe CSV com coluna 'ticker' (opcional)", type=["csv"])
    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            col = next((c for c in csv_df.columns if c.lower() == "ticker"), csv_df.columns[0])
            tickers_text = ", ".join(csv_df[col].dropna().astype(str).tolist())
        except Exception as exc:
            st.error(f"Não foi possível ler o CSV: {exc}")

    freq_choice = st.radio("Frequência", ["Mensal", "Diária", "Ambas"], index=2)
    start_date = st.date_input("Data inicial", value=date(2021, 1, 1), min_value=date(2000, 1, 1))
    end_date = st.date_input("Data final", value=date.today(), min_value=date(2000, 1, 1))
    frontier_points = st.slider("Pontos da fronteira", 30, 120, 70, 10)
    max_weight = st.slider("Peso máximo por ativo", 5, 100, 100, 5) / 100
    st.caption("Vendas a descoberto desabilitadas: pesos entre 0% e o limite definido.")
    run = st.button("Calcular carteiras", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    tickers = list(dict.fromkeys(re.findall(r"[A-Z0-9]{4}\d{1,2}", tickers_text.upper())))
    if len(tickers) < 2:
        st.error("Informe pelo menos dois tickers válidos (ex.: ITUB4, WEGE3).")
    elif max_weight * len(tickers) < 1 - 1e-9:
        st.error("O limite máximo por ativo é incompatível com a quantidade de ativos: a soma dos limites precisa permitir 100% de alocação.")
    elif start_date >= end_date:
        st.error("A data inicial precisa ser anterior à data final.")
    else:
        freqs = ["Mensal", "Diária"] if freq_choice == "Ambas" else [freq_choice]
        all_results = {}
        for freq in freqs:
            with st.spinner(f"Baixando dados e otimizando carteiras ({freq.lower()})..."):
                try:
                    prices = download_prices(tuple(tickers), start_date, end_date, freq)
                    if prices.shape[1] < 2:
                        raise ValueError("Menos de duas ações retornaram dados. Verifique os tickers.")
                    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
                    # Universo comum de observações: evita covariâncias com amostras distintas.
                    returns = returns.dropna(axis=1, how="any")
                    if returns.shape[1] < 2 or len(returns) < 20:
                        raise ValueError("Amostra insuficiente após alinhar os dados. São necessárias ao menos 20 observações e dois ativos completos.")
                    assets = returns.columns.tolist()
                    factor = annualization_factor(freq)
                    mu = returns.mean().values * factor
                    cov = returns.cov().values * factor
                    cov = (cov + cov.T) / 2
                    if np.linalg.eigvalsh(cov).min() < -1e-8:
                        raise ValueError("A matriz de covariância não é semidefinida positiva. Revise os dados.")
                    selic = download_selic(start_date, end_date, freq)
                    if freq == "Mensal":
                        rf = float(selic.mean() * 12)  # mesma convenção da planilha
                    else:
                        rf = float((1 + selic.mean()) ** 252 - 1)
                    w_mvp = optimize_mvp(mu, cov, max_weight)
                    w_tan = optimize_max_sharpe(mu, cov, rf, max_weight)
                    frontier = efficient_frontier(mu, cov, frontier_points, max_weight)
                    asset_stats = pd.DataFrame({
                        "Ativo": assets,
                        "Retorno esperado": mu,
                        "Volatilidade": np.sqrt(np.diag(cov))
                    }).sort_values("Retorno esperado", ascending=False)
                    all_results[freq] = {
                        "prices": prices[assets], "returns": returns[assets],
                        "assets": assets, "mu": mu, "cov": cov, "rf": rf,
                        "selic": selic, "mvp": w_mvp, "tangent": w_tan,
                        "frontier": frontier, "asset_stats": asset_stats,
                        "factor": factor
                    }
                except Exception as exc:
                    st.error(f"Erro na análise {freq}: {exc}")
        st.session_state.results = all_results if all_results else None

results = st.session_state.results
if not results:
    st.info("Configure o universo, o período e clique em **Calcular carteiras** para iniciar.")
    st.subheader("O que esta versão calcula")
    c1, c2, c3 = st.columns(3)
    c1.metric("Otimização", "Long-only", "Pesos não negativos")
    c2.metric("Frequências", "Diária e mensal", "Anualização 252 / 12")
    c3.metric("Taxa livre de risco", "Selic", "Banco Central do Brasil")
    st.markdown("""
    **Metodologia:** retornos simples sobre preços ajustados; média histórica para retorno esperado;
    covariância amostral anualizada; otimização de variância mínima e máximo Sharpe.
    """)
    st.stop()

tabs = st.tabs(list(results.keys()))
for tab, (freq, data) in zip(tabs, results.items()):
    with tab:
        assets = data["assets"]
        mu, cov, rf = data["mu"], data["cov"], data["rf"]
        w_mvp, w_tan = data["mvp"], data["tangent"]
        mvp_ret, mvp_vol = portfolio_stats(w_mvp, mu, cov)
        tan_ret, tan_vol = portfolio_stats(w_tan, mu, cov)
        sharpe = (tan_ret - rf) / tan_vol if tan_vol else np.nan

        st.subheader(f"Resultados — frequência {freq.lower()}")
        st.caption(f"{len(assets)} ativos • {len(data['returns'])} observações • anualização por {data['factor']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Retorno esperado — máximo Sharpe", pct(tan_ret))
        k2.metric("Volatilidade — máximo Sharpe", pct(tan_vol))
        k3.metric("Sharpe máximo", f"{sharpe:.3f}")
        k4.metric("Taxa livre de risco anual", pct(rf))

        frontier = data["frontier"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frontier["Volatilidade"], y=frontier["Retorno"],
                                 mode="lines", name="Fronteira eficiente",
                                 line=dict(width=3, color="#16A36A")))
        asset_vol = np.sqrt(np.diag(cov))
        fig.add_trace(go.Scatter(x=asset_vol, y=mu, mode="markers+text",
                                 text=assets, textposition="top center",
                                 name="Ativos", marker=dict(size=9, color="#3978C5")))
        fig.add_trace(go.Scatter(x=[mvp_vol], y=[mvp_ret], mode="markers",
                                 name="Mínima variância", marker=dict(size=13, symbol="diamond", color="#E09B32")))
        fig.add_trace(go.Scatter(x=[tan_vol], y=[tan_ret], mode="markers",
                                 name="Máximo Sharpe", marker=dict(size=14, symbol="star", color="#B34DCC")))
        fig.update_layout(
            template="plotly_white",
            font=dict(
                family="Inter",
                color="#526575",
                size=11
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=15, r=15, t=25, b=15)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Carteiras ótimas")
        left, right = st.columns(2)
        with left:
            st.markdown("**Carteira de mínima variância (MVP)**")
            st.metric("Retorno esperado", pct(mvp_ret))
            st.metric("Volatilidade", pct(mvp_vol))
            mvp_weights = pd.DataFrame({"Ativo": assets, "Peso": w_mvp}).sort_values("Peso", ascending=False)
            st.dataframe(mvp_weights.style.format({"Peso": "{:.2%}"}), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Carteira de máximo Sharpe**")
            st.metric("Retorno esperado", pct(tan_ret))
            st.metric("Volatilidade", pct(tan_vol))
            tan_weights = pd.DataFrame({"Ativo": assets, "Peso": w_tan}).sort_values("Peso", ascending=False)
            st.dataframe(tan_weights.style.format({"Peso": "{:.2%}"}), use_container_width=True, hide_index=True)

        pie = go.Figure(go.Pie(labels=assets, values=w_tan, hole=.45, sort=False))
        pie.update_layout(title="Composição da carteira de máximo Sharpe", height=430)
        st.plotly_chart(pie, use_container_width=True)

        st.markdown("#### Indicadores dos ativos")
        st.dataframe(data["asset_stats"].style.format({
            "Retorno esperado": "{:.2%}", "Volatilidade": "{:.2%}"
        }), use_container_width=True, hide_index=True)

        st.markdown("#### Matrizes de risco")
        corr = data["returns"].corr()
        cov_df = pd.DataFrame(cov, index=assets, columns=assets)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Matriz de correlação")
            st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
        with c2:
            st.caption("Matriz de covariância anualizada")
            st.dataframe(cov_df.style.format("{:.5f}"), use_container_width=True)

        st.markdown("#### Histórico de preços ajustados")
        price_fig = go.Figure()
        for asset in assets:
            normalized = data["prices"][asset] / data["prices"][asset].dropna().iloc[0] * 100
            price_fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=asset))
        price_fig.update_layout(yaxis_title="Índice (base 100)", height=400, hovermode="x unified")
        st.plotly_chart(price_fig, use_container_width=True)

        # Beta histórico contra BOVA11, quando o benchmark foi incluído.
        if "BOVA11" in assets:
            benchmark = data["returns"]["BOVA11"]
            betas = {}
            for asset in assets:
                if asset == "BOVA11":
                    continue
                aligned = pd.concat([data["returns"][asset], benchmark], axis=1).dropna()
                betas[asset] = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
            if betas:
                st.markdown("#### Beta em relação ao BOVA11")
                st.dataframe(pd.DataFrame({"Ativo": list(betas.keys()), "Beta": list(betas.values())})
                             .style.format({"Beta": "{:.3f}"}), hide_index=True, use_container_width=True)

        export = pd.DataFrame({
            "Ativo": assets,
            "Peso_MVP": w_mvp,
            "Peso_Maximo_Sharpe": w_tan,
            "Retorno_Esperado_Anual": mu,
            "Volatilidade_Anual": asset_vol
        })
        st.download_button(
            f"Baixar pesos e indicadores ({freq.lower()})",
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"markowitz_{freq.lower()}.csv",
            mime="text/csv"
        )

st.divider()
st.caption("Fontes: Yahoo Finance via yfinance (preços ajustados) e Banco Central do Brasil/SGS (Selic). "
           "Resultados históricos são estimativas e não constituem recomendação de investimento.")

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050))
    )
