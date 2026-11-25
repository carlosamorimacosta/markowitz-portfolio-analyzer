"""Markowitz Portfolio Analytics — Dash application for Render."""
import os
import re
import io
import base64
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import minimize
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc


BASE_DIR = Path(__file__).resolve().parent

app = Dash(
    __name__,
    assets_folder=str(BASE_DIR / "assets"),
    assets_url_path="assets",
    include_assets_files=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Markowitz | Portfolio Analytics"
)

server = app.server


@server.route('/health')
def health():
    return 'OK', 200

NAVY = "#10283F"
GREEN = "#16836B"
WHITE = "#FFFFFF"
BACKGROUND = "#F4F6F8"
BORDER = "#E1E6EC"
GRAY = "#64748B"
RED = "#B54747"

COLOR_PALETTE = [
    "#10283F",
    "#16836B",
    "#497B98",
    "#86B4A9",
    "#8295A8",
    "#B78127",
    "#B54747",
    "#CED9E2"
]

institutional_template = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="Inter, Arial, sans-serif",
            size=12,
            color=NAVY
        ),

        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,

        colorway=COLOR_PALETTE,

        title=dict(
            font=dict(
                size=16,
                color=NAVY
            ),
            x=0.02
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor=BORDER,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(
                size=11,
                color=GRAY
            ),
            automargin=True
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor=BORDER,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(
                size=11,
                color=GRAY
            ),
            automargin=True
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),

        hoverlabel=dict(
            bgcolor=NAVY,
            font_color=WHITE,
            font_family="Inter, Arial"
        ),

        margin=dict(
            l=55,
            r=25,
            t=65,
            b=50
        )
    )
)

pio.templates["institutional"] = institutional_template
pio.templates.default = "institutional"


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
        prices = prices.groupby(prices.index.to_period("M")).last()
        prices.index = prices.index.to_timestamp("M")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(axis=1, how="all")

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
    # Retorno máximo factível sob o limite individual.
    remaining = 1.0
    max_return = 0.0
    for j in np.argsort(mu)[::-1]:
        allocation = min(max_weight, remaining)
        max_return += allocation * mu[j]
        remaining -= allocation
        if remaining <= 1e-10:
            break
    targets = np.linspace(r_mvp, max(r_mvp, max_return), n_points)
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


# ---- Interface helpers ----
def kpi_card(label, value, description="", positive=False):
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value" + (" positive" if positive else "")),
        html.Div(description, className="kpi-description")
    ], className="kpi-card")


def panel(title, content, subtitle=None):
    return html.Div([html.H3(title, className="chart-title"),
                     html.Div(subtitle or "", className="chart-subtitle"), content],
                    className="chart-card")


def graph(fig):
    return dcc.Graph(figure=fig, config={"displaylogo": False, "responsive": True}, responsive=True)


def table(frame, formats=None):
    display = frame.copy()
    for col, fmt in (formats or {}).items():
        display[col] = display[col].map(lambda x: fmt.format(x) if pd.notna(x) else "—")
    return html.Div(dbc.Table.from_dataframe(display, striped=False, bordered=False,
                    hover=True, index=False, className="financial-table"), className="table-container")


def field(label, component):
    return html.Div([html.Label(label, className="input-label"), component])


sidebar = html.Aside([
    html.Div([html.H2("MARKOWITZ", className="brand-title"),
              html.Div("Portfolio Analytics", className="brand-subtitle")], className="sidebar-brand"),
    html.Div("Workspace", className="sidebar-section-title"),
    html.A("Parâmetros", href="#parameters", className="nav-link"),
    html.A("Resultados", href="#results", className="nav-link"),
    html.A("Metodologia", href="#methodology", className="nav-link"),
    html.Div(["Quantitative Research", html.Br(), "Mean-variance optimization"], className="sidebar-footer")
], className="sidebar")

filters = html.Div([
    html.H3("Parâmetros de investimento", className="section-title"),
    field("Universo de ativos", dcc.RadioItems(id="universe", options=[
        {"label": " Lista inicial editável", "value": "starter"},
        {"label": " Personalizado", "value": "custom"}], value="custom", inline=True)),
    html.P("A lista inicial não corresponde necessariamente à composição vigente do Ibovespa.", className="form-text"),
    field("Tickers separados por vírgula", dcc.Textarea(id="tickers", value="ITUB4, WEGE3, PETR4, VALE3",
            className="form-control", style={"width":"100%", "height":"90px"})),
    dcc.Upload(id="upload", children=html.Div("Importar CSV com coluna ticker"), className="upload-area", multiple=False),
    html.Div(id="upload-message"),
    html.Div([
        field("Período", dcc.DatePickerRange(id="period", start_date="2021-01-01", end_date=date.today(),
              min_date_allowed=date(2000,1,1), max_date_allowed=date.today(), display_format="DD/MM/YYYY")),
        field("Frequência", dcc.Dropdown(id="frequency", options=[{"label": x, "value": x} for x in ["Mensal", "Diária", "Ambas"]], value="Ambas", clearable=False)),
        field("Peso máximo por ativo (%)", dcc.Slider(id="max-weight", min=5, max=100, step=5, value=100,
                marks={5:"5",50:"50",100:"100"}, tooltip={"placement":"bottom"})),
        field("Pontos da fronteira", dcc.Slider(id="points", min=30,max=120,step=10,value=70,
                marks={30:"30",70:"70",120:"120"}))
    ], className="filter-grid", style={"marginTop":"20px"}),
    html.P("Estratégia long-only, integralmente investida, sem alavancagem.", className="form-text"),
    html.Button("Calcular carteiras", id="run", n_clicks=0, className="primary-button")
], id="parameters", className="filter-card")


# =========================================================
# APPLICATION LAYOUT
# =========================================================

app.layout = html.Div(
    [
        # Sidebar institucional
        sidebar,

        # Área principal
        html.Main(
            [
                # Cabeçalho superior
                html.Header(
                    [
                        html.Div(
                            "Workspace / Portfolio Analytics",
                            className="breadcrumb-text"
                        ),

                        html.Div(
                            [
                                html.Span(className="status-dot"),
                                html.Span("Brazilian Equities")
                            ],
                            className="status-indicator"
                        )
                    ],
                    className="topbar"
                ),

                # Conteúdo
                html.Div(
                    [
                        # Título da página
                        html.Div(
                            [
                                html.H1(
                                    "Portfolio Analytics",
                                    className="page-title"
                                ),

                                html.P(
                                    "Construção de carteiras e "
                                    "análise quantitativa de risco",
                                    className="page-subtitle"
                                )
                            ],
                            className="page-header"
                        ),

                        # Parâmetros de investimento
                        filters,

                        # Resultados dinâmicos
                        dcc.Loading(
                            id="loading-results",
                            type="circle",
                            color="#16836B",
                            children=html.Div(
                                id="results",
                                children=[
                                    html.Div(
                                        [
                                            html.H3(
                                                "Portfolio Overview",
                                                className="section-title"
                                            ),

                                            html.Div(
                                                [
                                                    kpi_card(
                                                        "Retorno esperado",
                                                        "—",
                                                        "Retorno anualizado"
                                                    ),

                                                    kpi_card(
                                                        "Volatilidade",
                                                        "—",
                                                        "Risco anualizado"
                                                    ),

                                                    kpi_card(
                                                        "Índice de Sharpe",
                                                        "—",
                                                        "Retorno ajustado ao risco"
                                                    ),

                                                    kpi_card(
                                                        "Taxa livre de risco",
                                                        "—",
                                                        "Selic histórica"
                                                    )
                                                ],
                                                className="kpi-grid"
                                            ),

                                            dbc.Alert(
                                                "Configure os parâmetros "
                                                "de investimento e clique "
                                                "em Calcular carteiras.",
                                                color="info"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ),

                        # Metodologia
                        html.Section(
                            [
                                html.H2(
                                    "Metodologia",
                                    className="section-title"
                                ),

                                html.P(
                                    "O modelo utiliza a abordagem "
                                    "de média-variância proposta "
                                    "por Harry Markowitz (1952)."
                                ),

                                html.H3(
                                    "Construção de carteiras"
                                ),

                                html.P(
                                    "Os retornos esperados são "
                                    "estimados pela média histórica "
                                    "dos retornos simples. A matriz "
                                    "de covariância amostral é "
                                    "anualizada de acordo com a "
                                    "frequência selecionada."
                                ),

                                html.H3(
                                    "Otimização"
                                ),

                                html.P(
                                    "As carteiras de mínima "
                                    "variância e máximo Sharpe são "
                                    "obtidas por otimização numérica "
                                    "via SLSQP, considerando pesos "
                                    "não negativos e investimento "
                                    "integral do patrimônio."
                                ),

                                html.H3(
                                    "Fontes de dados"
                                ),

                                html.P(
                                    "Yahoo Finance: preços "
                                    "históricos ajustados."
                                ),

                                html.P(
                                    "Banco Central do Brasil "
                                    "(SGS): taxa Selic histórica."
                                ),

                                html.H3(
                                    "Limitações"
                                ),

                                html.P(
                                    "Os resultados representam "
                                    "estimativas históricas e são "
                                    "sensíveis às hipóteses de "
                                    "retorno esperado e covariância. "
                                    "Não constituem recomendação "
                                    "de investimento."
                                )
                            ],
                            id="methodology",
                            className="methodology-card"
                        ),

                        # Rodapé
                        html.Footer(
                            [
                                html.Div(
                                    "MARKOWITZ | Portfolio Analytics"
                                ),

                                html.Div(
                                    "Yahoo Finance · Banco Central "
                                    "do Brasil · Quantitative Research"
                                )
                            ],
                            className="dashboard-footer"
                        )
                    ],
                    className="content-wrapper"
                )
            ],
            className="main-content"
        )
    ],
    className="app-container"
)

@app.callback(Output("tickers", "value"), Output("upload-message", "children"),
              Input("universe", "value"), Input("upload", "contents"), prevent_initial_call=True)
def change_universe(mode, contents):
    from dash import ctx
    if ctx.triggered_id == "upload" and contents:
        try:
            raw = base64.b64decode(contents.split(",", 1)[1])
            if len(raw) > 1_000_000:
                raise ValueError("CSV limitado a 1 MB.")
            frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
            col = next((c for c in frame if str(c).strip().lower() == "ticker"), frame.columns[0])
            return ", ".join(frame[col].dropna().astype(str)), dbc.Alert("CSV importado.", color="success")
        except Exception as exc:
            return no_update, dbc.Alert(f"Erro no CSV: {exc}", color="danger")
    return (", ".join(IBOV_STARTER[:20]) if mode == "starter" else "ITUB4, WEGE3, PETR4, VALE3"), ""


def analyze(tickers, start, end, freq, limit, points):
    prices = download_prices(tuple(tickers), start, end, freq)
    # Excluir mês incompleto; não preencher preços ausentes artificialmente.
    if freq == "Mensal":
        cutoff = min(pd.Timestamp(end), pd.Timestamp(date.today()))
        prices = prices.loc[prices.index <= cutoff]
        if cutoff.date() == date.today() and len(prices) and prices.index[-1].to_period('M') == cutoff.to_period('M'):
            prices = prices.iloc[:-1]
    returns = prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, how="any")
    if returns.shape[1] < 2 or len(returns) < 20:
        raise ValueError("Amostra insuficiente: ao menos 20 retornos e dois ativos completos.")
    assets = returns.columns.tolist()
    if len(assets)*limit < 1-1e-9:
        raise ValueError("Limite de peso incompatível com o universo efetivamente disponível.")
    factor = annualization_factor(freq)
    mu = returns.mean().to_numpy()*factor
    cov = returns.cov().to_numpy()*factor
    cov = (cov+cov.T)/2
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError("Matriz de covariância inválida.")
    selic = download_selic(start, end, freq).dropna()
    if freq == "Mensal":
        selic = selic[selic.index.to_period('M').isin(returns.index.to_period('M'))]
    else:
        selic = selic[selic.index.normalize().isin(returns.index.normalize())]
    if selic.empty:
        raise ValueError("Selic indisponível para as observações utilizadas.")
    rf = float(selic.mean()*12 if freq == "Mensal" else (1+selic.mean())**252-1)
    mvp = optimize_mvp(mu,cov,limit)
    tan = optimize_max_sharpe(mu,cov,rf,limit)
    frontier = efficient_frontier(mu,cov,points,limit)
    return assets,prices[assets],returns,mu,cov,rf,mvp,tan,frontier


def render_results(tickers, start, end, freq, limit, points):
    assets,prices,returns,mu,cov,rf,mvp,tan,frontier = analyze(tickers,start,end,freq,limit,points)
    mr,mv = portfolio_stats(mvp,mu,cov)
    tr,tv = portfolio_stats(tan,mu,cov)
    sharpe = (tr-rf)/tv if tv>1e-12 else np.nan
    av = np.sqrt(np.maximum(np.diag(cov),0))
    fig = go.Figure([
        go.Scatter(x=frontier.Volatilidade,y=frontier.Retorno,mode="lines",name="Fronteira eficiente",line=dict(color=NAVY,width=3)),
        go.Scatter(x=av,y=mu,mode="markers+text",text=assets,textposition="top center",name="Ativos",marker=dict(color="#8295A8",size=8)),
        go.Scatter(x=[mv],y=[mr],mode="markers",name="Mínima variância",marker=dict(color="#497B98",size=12,symbol="diamond")),
        go.Scatter(x=[tv],y=[tr],mode="markers",name="Máximo Sharpe",marker=dict(color=GREEN,size=14,symbol="star"))])
    fig.update_layout(xaxis_title="Volatilidade anualizada",yaxis_title="Retorno esperado anualizado",xaxis_tickformat=".1%",yaxis_tickformat=".1%",height=460)
    pie = go.Figure(go.Pie(labels=assets,values=np.maximum(tan,0),hole=.60,sort=False))
    pie.update_layout(height=460,showlegend=True)
    holdings = pd.DataFrame({"Ativo":assets,"Peso MVP":mvp,"Peso máximo Sharpe":tan,"Retorno esperado":mu,"Volatilidade":av})
    holdings = holdings.sort_values("Peso máximo Sharpe",ascending=False)
    formats={c:"{:.2%}" for c in holdings.columns if c!="Ativo"}
    corr=returns.corr()
    cov_df=pd.DataFrame(cov,index=assets,columns=assets)
    heat=go.Figure(go.Heatmap(z=corr.values,x=assets,y=assets,zmin=-1,zmax=1,colorscale=[[0,"#B54747"],[.5,"#FFFFFF"],[1,NAVY]],colorbar=dict(title="Correlação")))
    heat.update_layout(height=max(350,min(850,25*len(assets)+120)))
    price_fig=go.Figure()
    for asset in assets:
        series=prices[asset].dropna()
        price_fig.add_trace(go.Scatter(x=series.index,y=series/series.iloc[0]*100,name=asset,mode="lines"))
    price_fig.update_layout(yaxis_title="Índice (base 100)",height=420,hovermode="x unified")
    excluded=sorted(set(tickers)-set(assets))
    warnings=[dbc.Alert("Ativos excluídos por indisponibilidade ou dados incompletos: "+", ".join(excluded),color="warning")] if excluded else []
    beta_panel=[]
    # Benchmark separado do universo investível.
    try:
        bench=returns["BOVA11"] if "BOVA11" in returns else download_prices(("BOVA11",),start,end,freq)["BOVA11"].pct_change(fill_method=None)
        betas=[]
        for asset in assets:
            pair=pd.concat([returns[asset],bench],axis=1).dropna()
            variance=pair.iloc[:,1].var()
            if len(pair)>=20 and variance>0:
                betas.append({"Ativo":asset,"Beta":pair.iloc[:,0].cov(pair.iloc[:,1])/variance})
        if betas:
            beta_panel=[panel("Beta histórico vs. BOVA11",table(pd.DataFrame(betas),{"Beta":"{:.3f}"}))]
    except Exception:
        beta_panel=[dbc.Alert("Benchmark BOVA11 indisponível; os demais resultados foram preservados.",color="warning")]
    csv=holdings.to_csv(index=False).encode("utf-8-sig")
    href="data:text/csv;base64,"+base64.b64encode(csv).decode("ascii")
    matrix=cov_df.rename_axis("Ativo").reset_index()
    return html.Div([
        html.P(f"{len(assets)} ativos · {len(returns)} observações · fator de anualização {annualization_factor(freq)}",className="page-subtitle"),
        *warnings,
        html.Div([kpi_card("Retorno esperado",pct(tr),"Máximo Sharpe",tr>0),kpi_card("Volatilidade",pct(tv),"Anualizada"),kpi_card("Sharpe",f"{sharpe:.3f}" if np.isfinite(sharpe) else "—","Máximo Sharpe"),kpi_card("Taxa livre de risco",pct(rf),"Selic histórica anualizada")],className="kpi-grid"),
        html.Div([panel("Fronteira eficiente",graph(fig)),panel("Alocação ótima",graph(pie),"Máximo Sharpe")],className="chart-grid"),
        html.Div([panel("Mínima variância",html.P(f"Retorno: {pct(mr)} · Volatilidade: {pct(mv)}")),panel("Máximo Sharpe",html.P(f"Retorno: {pct(tr)} · Volatilidade: {pct(tv)}"))],className="equal-grid"),
        panel("Carteiras e indicadores dos ativos",table(holdings,formats)),
        html.Div(
            [
                panel(
                    "Matriz de correlação",
                    graph(heat)
                ),
        
                panel(
                    "Histórico de preços ajustados",
                    graph(price_fig)
                )
            ],
            className="equal-grid"
        ),
        
        panel(
            "Covariância anualizada",
            table(
                matrix,
                {c: "{:.5f}" for c in assets}
            )
        ),
        *beta_panel,
        html.A("Baixar pesos e indicadores (CSV)",href=href,download=f"markowitz_{freq.lower()}.csv",className="secondary-button")
    ])


@app.callback(Output("results","children"),Input("run","n_clicks"),State("tickers","value"),State("period","start_date"),State("period","end_date"),State("frequency","value"),State("max-weight","value"),State("points","value"),prevent_initial_call=True)
def run_optimization(n, tickers_text, start, end, frequency, max_weight, points):
    try:
        tickers=list(dict.fromkeys(re.findall(r"\b[A-Z]{4}\d{1,2}\b",(tickers_text or "").upper())))
        if len(tickers)<2 or len(tickers)>60:
            raise ValueError("Informe entre 2 e 60 tickers válidos.")
        start,end=date.fromisoformat(start[:10]),date.fromisoformat(end[:10])
        if start>=end or end>date.today():
            raise ValueError("Período inválido.")
        limit=max_weight/100
        if limit*len(tickers)<1-1e-9:
            raise ValueError("O limite de peso não permite 100% de alocação.")
        freqs=["Mensal","Diária"] if frequency=="Ambas" else [frequency]
        tabs=[]
        for freq in freqs:
            try:
                children=render_results(tickers,start,end,freq,limit,points)
            except Exception as exc:
                children=dbc.Alert(f"Erro na análise {freq}: {exc}",color="danger")
            tabs.append(dcc.Tab(label=freq,children=children))
        return html.Div([html.H2("Resultados da otimização",className="section-title"),dcc.Tabs(children=tabs)])
    except Exception as exc:
        return dbc.Alert(str(exc),color="danger")


if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",8050)))
