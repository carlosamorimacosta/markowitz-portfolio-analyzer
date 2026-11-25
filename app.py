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
from scipy.optimize import minimize, linprog, LinearConstraint
from scipy.stats import norm
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
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    # Use the same last observed trading session for all assets each month.
    if frequency == "Mensal":
        prices = prices.groupby(prices.index.to_period("M")).tail(1)
        prices.index = prices.index.to_period("M").to_timestamp("M")
    return prices.dropna(axis=1, how="all")


def download_selic(start, end, frequency):
    # SGS 4390: Selic acumulada no mês (% a.m.); SGS 11: Selic diária (% a.d.).
    code = 4390 if frequency == "Mensal" else 11
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    data = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=365*9))
        params = {"formato": "json", "dataInicial": cursor.strftime("%d/%m/%Y"),
                  "dataFinal": stop.strftime("%d/%m/%Y")}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data.extend(response.json())
        cursor = stop + timedelta(days=1)
    if not data:
        raise ValueError("O Banco Central não retornou observações da Selic para o período.")
    frame = pd.DataFrame(data)
    frame["data"] = pd.to_datetime(frame["data"], dayfirst=True)
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce") / 100
    series = frame.set_index("data")["valor"].sort_index()
    return series


def annualization_factor(frequency):
    return 12 if frequency == "Mensal" else 252


# ---- Portfolio optimization and risk engine ----


def optimize_portfolios(mu, cov, rf, points=70, limit=1., allow_short=False,
                        gross_limit=1., net_target=1.):
    """Fixed net risky exposure; residual cash earns/pays rf. Gross cap uses |w|.

    Auxiliary variables u >= |w| make gross constraints linear. The Sharpe
    search uses multiple starts, not a global guarantee in negative-excess cases.
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    n = len(mu)
    if n < 2 or not 0 < limit <= 3 or not 1 <= net_target <= gross_limit <= 3:
        raise ValueError('Exposição inválida: 100% <= líquida <= bruta <= 300%.')
    if net_target > n * limit + 1e-9:
        raise ValueError('Limite por ativo incompatível com a exposição líquida.')
    if not np.isfinite(mu).all() or not np.isfinite(cov).all() or not np.isfinite(rf):
        raise ValueError('Estimativas não finitas.')
    if cov.shape != (n,n) or np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError('Covariância inválida.')
    cov = (cov + cov.T) / 2
    I = np.eye(n)
    zero = np.zeros(n)
    A = np.vstack([np.hstack([I,-I]),np.hstack([-I,-I]),np.r_[zero,np.ones(n)]])
    b = np.r_[np.zeros(2*n),gross_limit]
    E = np.r_[np.ones(n),zero][None,:]
    bounds = [(-limit if allow_short else 0.,limit)]*n + [(0.,limit)]*n
    # For long-only or gross==net the auxiliary constraint is redundant;
    # removing it avoids degenerate active constraints at the solution.
    if not allow_short or abs(gross_limit-net_target)<1e-10:
        A = np.empty((0,n)); b = np.empty(0); E = np.ones((1,n))
        bounds = [(0.,limit)]*n; c = mu.copy()
        x0 = np.full(n,net_target/n)
    else:
        c = np.r_[mu,zero]; x0 = np.r_[np.full(n,net_target/n),np.full(n,net_target/n)]
    cash = (1-net_target)*rf
    constraints = [LinearConstraint(E, net_target, net_target)]
    if len(b): constraints.append(LinearConstraint(A,-np.inf,b))
    def feasible(x, target=None):
        w = x[:n]
        return (np.isfinite(x).all() and abs(w.sum()-net_target)<2e-6
                and np.abs(w).sum() <= gross_limit+2e-6
                and all(lo-2e-6 <= v <= hi+2e-6 for v,(lo,hi) in zip(x,bounds))
                and (not len(b) or np.max(A@x-b)<2e-6)
                and (target is None or abs(c@x+cash-target)<2e-6))
    def var(x): return float(x[:n]@cov@x[:n])
    scale = max(float(np.max(np.diag(cov))),1e-8)
    def var_obj(x): return var(x)/scale
    def var_grad(x): return np.r_[2*cov@x[:n]/scale,np.zeros(len(x)-n)]
    def sharpe(x):
        vol=np.sqrt(max(var(x),0.))
        return float((c@x+cash-rf)/vol) if vol>1e-10 else np.nan
    def solve(fun,x,extra=(),jac=None):
        return minimize(fun,x,jac=jac,method='SLSQP',bounds=bounds,
                        constraints=constraints+list(extra),options={'maxiter':1000,'ftol':1e-10})
    lp=linprog(-c,A_ub=A if len(b) else None,b_ub=b if len(b) else None,
               A_eq=E,b_eq=[net_target],bounds=bounds,method='highs')
    if not lp.success or not feasible(lp.x): raise ValueError('Restrições de exposição inviáveis.')
    m=solve(var_obj,x0,jac=var_grad)
    if not m.success or not feasible(m.x): raise RuntimeError('Falha ao calcular MVP: '+str(m.message))
    r_min=float(c@m.x+cash); r_max=max(r_min,float(c@lp.x+cash))
    candidates=[m.x,lp.x,x0]; rows=[]
    targets=np.linspace(r_min,r_max,max(2,int(points))) if r_max-r_min>1e-9 else [r_min]
    for target in targets:
        if abs(target-r_min)<1e-10: x=m.x
        else:
            frac=(target-r_min)/(r_max-r_min)
            seed=(1-frac)*m.x+frac*lp.x
            res=solve(var_obj,seed,[LinearConstraint(c[None,:],target-cash,target-cash)],jac=var_grad)
            if not res.success or not feasible(res.x,target): continue
            x=res.x
        rows.append({'Retorno':float(c@x+cash),'Volatilidade':np.sqrt(max(var(x),0.)),'Pesos':x[:n].copy()})
        candidates.append(x)
    frontier=pd.DataFrame(rows)
    valid=[x for x in candidates if feasible(x) and np.isfinite(sharpe(x))]
    if not valid: raise ValueError('Sharpe indefinido: volatilidade nula.')
    best=max(valid,key=sharpe)
    # Include the best grid point, the MVP, the max-return point and equal weights.
    for seed in (best,m.x,lp.x,x0):
        opt=solve(lambda x:-sharpe(x) if np.isfinite(sharpe(x)) else 1e8,seed)
        if opt.success and feasible(opt.x) and np.isfinite(sharpe(opt.x)) and sharpe(opt.x)>sharpe(best): best=opt.x
    return m.x[:n],best[:n],frontier




def portfolio_return_series(returns, weights, cash_returns=None):
    w=np.asarray(weights,float)
    r=returns@w
    cash=1-w.sum()
    if abs(cash)>1e-8:
        if cash_returns is None: raise ValueError('Selic necessária para financiar exposição líquida acima de 100%.')
        rates=cash_returns.reindex(r.index)
        if rates.isna().any(): raise ValueError('Selic incompleta na amostra da carteira alavancada.')
        r=r+cash*rates
    return r.rename('Carteira')




def portfolio_moments(weights,mu,cov,rf):
    w=np.asarray(weights,float)
    return float(w@mu+(1-w.sum())*rf),float(np.sqrt(max(w@cov@w,0.)))




def var_metrics(r, levels=(.90,.95,.99)):
    r=pd.Series(r).dropna()
    if len(r)<2: raise ValueError('Retornos insuficientes para VaR.')
    return {f'VaR {int(c*100)}% {kind}': float(v)
            for c in levels for kind,v in [('histórico',-(r.quantile(1-c))),('paramétrico',norm.ppf(c)*r.std(ddof=1)-r.mean())]}




def wealth_metrics(r):
    """Starting wealth=1; stop at insolvency (no negative-NAV compounding)."""
    r=pd.Series(r).dropna().astype(float)
    if r.empty: raise ValueError('Retornos insuficientes para desempenho.')
    ruined=np.flatnonzero(r.to_numpy()<=-1)
    insolvency=bool(len(ruined))
    if insolvency:
        r=r.iloc[:ruined[0]+1].copy(); r.iloc[-1]=-1.
    wealth=(1+r).cumprod()
    # Include initial wealth so the first return is counted in drawdown/run-up.
    running_peak=wealth.cummax().clip(lower=1.)
    running_low=wealth.cummin().clip(upper=1.)
    dd=wealth/running_peak-1
    ru=(wealth/running_low.replace(0,np.nan)-1).fillna(0.)
    return {'Retorno acumulado':float(wealth.iloc[-1]-1),
            'Maximum Drawdown':float(min(0.,dd.min())),'Maximum Run-up':float(max(0.,ru.max())),
            'Insolvência':insolvency,'wealth':wealth,'drawdown':dd,'runup':ru}




def beta_against_market(r, market):
    pair=pd.concat([r.rename('p'),market.rename('m')],axis=1).dropna()
    if len(pair)<20 or pair.m.var()<=1e-16: return np.nan
    return float(pair.p.cov(pair.m)/pair.m.var())




def risk_contribution(weights, cov, assets):
    w=np.asarray(weights,float); cov=np.asarray(cov,float)
    variance=float(w@cov@w)
    if variance<=1e-16:
        return pd.DataFrame({'Ativo':assets,'MCR':np.nan,'CR':np.nan,'PCR':np.nan})
    marginal=cov@w/np.sqrt(variance)
    contribution=w*marginal
    return pd.DataFrame({'Ativo':assets,'MCR':marginal,'CR':contribution,'PCR':contribution/np.sqrt(variance)})




def exposure(weights):
    w=np.asarray(weights,float)
    return {'Long':float(np.maximum(w,0).sum()),'Short':float(-np.minimum(w,0).sum()),
            'Exposição líquida':float(w.sum()),'Exposição bruta':float(np.abs(w).sum()),'Caixa / financiamento':float(1-w.sum())}




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
                    className="chart-card", style={"height":"auto", "minWidth":0, "overflow":"visible"})


def graph(fig, height=None):
    if height is None:
        height = int(fig.layout.height or 420)
    fig.update_layout(
        autosize=True,
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(
            l=65,
            r=30,
            t=55,
            b=110 if fig.layout.legend.y is not None and fig.layout.legend.y < 0 else 70
        ),
        font=dict(
            family="Inter, Arial, sans-serif",
            size=12,
            color="#10283F"
        )
    )
    return dcc.Graph(
        figure=fig,
        className="institutional-graph",
        style={
            "width": "100%",
            "height": f"{height}px",
            "minHeight": f"{height}px",
            "flexShrink": 0
        },
        config={
            "displaylogo": False,
            "responsive": True,
            "displayModeBar": "hover",
            "scrollZoom": False,
            "toImageButtonOptions": {
                "format": "png",
                "scale": 2
            }
        },
        responsive=True
    )


def table(frame, formats=None):
    display = frame.copy()
    for col, fmt in (formats or {}).items():
        display[col] = display[col].map(lambda x: fmt.format(x) if pd.notna(x) else "—")
    return html.Div(dbc.Table.from_dataframe(display, striped=False, bordered=False,
                    hover=True, index=False, className="financial-table"), className="table-container")


def field(label, component):
    return html.Div([html.Label(label, className="input-label"), component])
sidebar = html.Aside(
    [
        # Identidade visual
        html.Div(
            [
                html.H2(
                    "MARKOWITZ",
                    className="brand-title"
                ),
                html.Div(
                    "Portfolio Analytics",
                    className="brand-subtitle"
                )
            ],
            className="sidebar-brand"
        ),
        # Navegação
        html.Div(
            "Workspace",
            className="sidebar-section-title"
        ),
        html.A(
            "Parâmetros",
            href="#parameters",
            className="nav-link"
        ),
        html.A(
            "Resultados",
            href="#results",
            className="nav-link"
        ),
        html.A(
            "Metodologia",
            href="#methodology",
            className="nav-link"
        ),
        # Rodapé da sidebar
        html.Div(
            [
                html.Div(
                    "Quantitative Research",
                    className="sidebar-research"
                ),
                html.Div(
                    "Mean-variance optimization",
                    className="sidebar-research-subtitle"
                ),
                # Identificação do desenvolvedor
                html.Div(
                    [
                        html.Div(
                            "DEVELOPED BY",
                            className="author-label"
                        ),
                        html.Div(
                            "Carlos Henrique",
                            className="author-name"
                        ),
                        html.Div(
                            "Economics Student",
                            className="author-affiliation"
                        ),
                        html.Div(
                            "FGV EESP",
                            className="author-affiliation"
                        ),
                        # Links profissionais
                        html.Div(
                            [
                                html.A(
                                    "GitHub ↗",
                                    href="https://github.com/carlosamorimacosta",
                                    target="_blank",
                                    rel="noopener noreferrer"
                                ),
                                html.A(
                                    "LinkedIn ↗",
                                    href="https://www.linkedin.com/in/carloshenriqueamorim",
                                    target="_blank",
                                    rel="noopener noreferrer"
                                )
                            ],
                            className="author-links"
                        )
                    ],
                    className="author-signature"
                )
            ],
            className="sidebar-footer"
        )
    ],
    className="sidebar"
)
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
        field("Frequência", dcc.Dropdown(id="frequency", options=[{"label": x, "value": x} for x in ["Mensal", "Diária", "Ambas"]], value="Mensal", clearable=False)),
        field("Peso absoluto máximo por ativo (%)", dcc.Slider(id="max-weight", min=5, max=100, step=5, value=100,
                marks={5:"5",50:"50",100:"100"}, tooltip={"placement":"bottom"})),
        field("Pontos da fronteira", dcc.Slider(id="points", min=30,max=120,step=10,value=70,
                marks={30:"30",70:"70",120:"120"}))
    ], className="filter-grid", style={"marginTop":"20px"}),
    html.Div([
        field("Vendas a descoberto", dcc.RadioItems(id="allow-short", options=[
            {"label":" Não", "value":"no"}, {"label":" Sim", "value":"yes"}],
            value="no", inline=True)),
        field("Exposição líquida (%)", dcc.Slider(id="net-target", min=100, max=300,
            step=10, value=100, marks={100:"100%",200:"200%",300:"300%"}, tooltip={"placement":"bottom", "always_visible":True})),
        field("Exposição bruta máxima (%)", dcc.Slider(id="gross-limit", min=100, max=300,
            step=10, value=100, marks={100:"100%",200:"200%",300:"300%"}, tooltip={"placement":"bottom", "always_visible":True}))
    ], className="filter-grid", style={"marginTop":"18px"}),
    html.P("Exposição líquida não pode exceder a bruta. Peso máximo é absoluto por ativo. "
           "Financiamento residual estimado pela Selic; sem custos de aluguel ou margem.", className="form-text"),
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
                                    "via SLSQP, respeitando limites individuais, "
                                    "exposição líquida e exposição bruta. "
                                    "Posições vendidas são opcionais."
                                ),
                                html.H3("Risco, desempenho e financiamento"),
                                html.P("VaR histórico (quantil empírico de perda) e paramétrico normal, "
                                       "em 90%, 95% e 99%, no horizonte da frequência escolhida. "
                                       "Um VaR negativo representa um ganho no quantil estimado, não uma perda."),
                                html.P("Retorno acumulado, drawdown e run-up são calculados com patrimônio inicial "
                                       "igual a 100 e pesos constantes, rebalanceados a cada período. "
                                       "Os pesos são estimados na mesma amostra: análise in-sample, não backtest."),
                                html.P("Beta: covariância com o BOVA11 dividida pela variância do benchmark. "
                                       "PCR: contribuição assinada para a volatilidade, pela decomposição de Euler; "
                                       "soma de 100% quando a variância é positiva."),
                                html.P("O caixa residual é 1 menos a exposição líquida. Financiamento à Selic histórica; "
                                       "retorno esperado e Selic anualizados aritmeticamente (12 ou 252). "
                                       "Otimização e PCR tratam a taxa de financiamento como determinística; "
                                       "o VaR e o desempenho usam a Selic realizada em cada período. "
                                       "Não são modelados custos de aluguel, spreads, impostos, transações ou chamadas de margem."),
                                html.P("Exposição líquida de 100% a 300%, bruta de até 300%, com limite individual absoluto. "
                                       "Otimização de Sharpe com múltiplos pontos iniciais, sem garantia global em todos os casos."),
                                html.H3("Fontes de dados"),
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
                                html.Div("Developed by Carlos Henrique · Economics Student, FGV EESP"),
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


def analyze(tickers, start, end, freq, limit, points, allow_short=False, gross_limit=1., net_target=1.):
    prices = download_prices(tuple(tickers), start, end, freq)
    if freq == "Mensal":
        cutoff = min(pd.Timestamp(end), pd.Timestamp(date.today()))
        prices = prices.loc[prices.index <= cutoff]
        prices = prices.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq=pd.offsets.MonthEnd())) if len(prices) else prices
    returns = prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).iloc[1:]
    returns = returns.dropna(axis=1, how="any")
    if returns.shape[1] < 2 or len(returns) < 20:
        raise ValueError("Amostra insuficiente: ao menos 20 retornos e dois ativos completos.")
    assets = returns.columns.tolist()
    factor = annualization_factor(freq)
    mu = returns.mean().to_numpy()*factor
    cov = returns.cov().to_numpy()*factor
    cov = (cov+cov.T)/2
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError("Matriz de covariância inválida.")
    # Monthly SGS dates refer to the first day of the reference month.
    selic = download_selic(start.replace(day=1) if freq == "Mensal" else start, end, freq).dropna()
    selic.index = (selic.index.to_period('M').to_timestamp('M') if freq == 'Mensal' else selic.index.normalize())
    selic = selic.groupby(level=0).last().reindex(returns.index)
    if selic.dropna().empty:
        raise ValueError("Selic indisponível para as observações utilizadas.")
    if selic.isna().any():
        raise ValueError("Selic incompleta nas datas dos retornos; ajuste o período ou tente novamente.")
    # Arithmetic annualization consistent with annualized arithmetic expected returns.
    rf = float(selic.mean()*factor)
    mvp,tan,frontier = optimize_portfolios(mu,cov,rf,points,limit,allow_short,gross_limit,net_target)
    return assets,prices[assets],returns,mu,cov,rf,mvp,tan,frontier,selic



def render_results(tickers, start, end, freq, limit, points, allow_short=False, gross_limit=1., net_target=1.):
    assets,prices,returns,mu,cov,rf,mvp,tan,frontier,selic = analyze(tickers,start,end,freq,limit,points,allow_short,gross_limit,net_target)
    mr,mv = portfolio_moments(mvp,mu,cov,rf)
    tr,tv = portfolio_moments(tan,mu,cov,rf)
    sharpe = (tr-rf)/tv if tv>1e-12 else np.nan
    av = np.sqrt(np.maximum(np.diag(cov),0))
    fig = go.Figure([
        go.Scatter(x=frontier.Volatilidade,y=frontier.Retorno,mode="lines",name="Fronteira eficiente",line=dict(color=NAVY,width=3)),
        go.Scatter(x=av,y=mu,mode="markers+text",text=assets,textposition="top center",name="Ativos",marker=dict(color="#8295A8",size=8)),
        go.Scatter(x=[mv],y=[mr],mode="markers",name="Mínima variância",marker=dict(color="#497B98",size=12,symbol="diamond")),
        go.Scatter(x=[tv],y=[tr],mode="markers",name="Máximo Sharpe",marker=dict(color=GREEN,size=14,symbol="star"))])
    fig.update_layout(
        height=440,
        xaxis_title="Volatilidade anualizada",
        yaxis_title="Retorno esperado anualizado",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode="closest"
    )
    allocation = go.Figure()
    allocation.add_bar(x=assets,y=mvp,name="Mínima variância",marker_color="#497B98")
    allocation.add_bar(x=assets,y=tan,name="Máximo Sharpe",marker_color=GREEN)
    allocation.update_layout(height=440,barmode="group",yaxis_tickformat=".0%",yaxis_title="Peso sobre patrimônio",xaxis_tickangle=-45)
    allocation.update_yaxes(zeroline=True,zerolinecolor=GRAY)

    holdings = pd.DataFrame({"Ativo":assets,"Peso MVP":mvp,"Peso máximo Sharpe":tan,"Retorno esperado":mu,"Volatilidade":av})
    holdings = holdings.sort_values("Peso máximo Sharpe",ascending=False)
    formats={c:"{:.2%}" for c in holdings.columns if c!="Ativo"}
    corr=returns.corr()
    cov_df=pd.DataFrame(cov,index=assets,columns=assets)
    heat=go.Figure(go.Heatmap(z=corr.values,x=assets,y=assets,zmin=-1,zmax=1,colorscale=[[0,"#B54747"],[.5,"#FFFFFF"],[1,NAVY]],colorbar=dict(title="Correlação")))
    heat.update_layout(
        height=max(
            420,
            min(850, 30 * len(assets) + 150)
        ),
        xaxis=dict(
            side="bottom",
            tickangle=-45
        ),
        yaxis=dict(
            autorange="reversed"
        )
    )
    price_fig=go.Figure()
    for asset in assets:
        series=prices[asset].dropna()
        price_fig.add_trace(go.Scatter(x=series.index,y=series/series.iloc[0]*100,name=asset,mode="lines"))
    price_fig.update_layout(
        height=420,
        xaxis_title="Período",
        yaxis_title="Índice (base 100)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )
    excluded=sorted(set(tickers)-set(assets))
    warnings=[dbc.Alert("Ativos excluídos por indisponibilidade ou dados incompletos: "+", ".join(excluded),color="warning")] if excluded else []
    if len(frontier) < points and len(frontier) > 1:
        warnings.append(dbc.Alert(f"Fronteira: {len(frontier)} de {points} pontos convergiram.", color="warning"))
    benchmark = None
    try:
        if "BOVA11" in returns:
            benchmark = returns["BOVA11"].copy()
        else:
            bp = download_prices(("BOVA11",),start,end,freq)["BOVA11"]
            if freq == "Mensal" and len(bp):
                bp=bp.reindex(pd.date_range(bp.index.min(),bp.index.max(),freq=pd.offsets.MonthEnd()))
            benchmark = bp.reindex(prices.index).pct_change(fill_method=None).reindex(returns.index)
        if benchmark.dropna().empty:
            benchmark=None
    except Exception:
        benchmark=None
    if benchmark is None or benchmark.dropna().size < 20:
        warnings.append(dbc.Alert("BOVA11 indisponível ou amostra insuficiente para beta.",color="warning"))
    if len(returns) < 500:
        warnings.append(dbc.Alert("VaR histórico de 99%: menos de 500 observações (menos de cinco observações esperadas na cauda de 1%); estimativa pouco precisa.",color="warning"))
    strategies={"Mínima variância":mvp,"Máximo Sharpe":tan}
    metrics={}; contributions={}
    wealth_fig=go.Figure(); drawdown_fig=go.Figure(); runup_fig=go.Figure(); pcr_fig=go.Figure()
    colors={"Mínima variância":"#497B98","Máximo Sharpe":GREEN}
    for name,w in strategies.items():
        r=portfolio_return_series(returns,w,selic)
        wealth=wealth_metrics(r)
        er,vol=portfolio_moments(w,mu,cov,rf)
        values={"Retorno esperado anualizado":er,"Volatilidade anualizada":vol,
                "Sharpe":(er-rf)/vol if vol>1e-10 else np.nan}
        values.update(var_metrics(r))
        values.update({k:wealth[k] for k in ("Retorno acumulado","Maximum Drawdown","Maximum Run-up")})
        values["Beta vs. BOVA11"]=beta_against_market(r,benchmark) if benchmark is not None else np.nan
        values.update(exposure(w))
        metrics[name]=values
        rc=risk_contribution(w,cov,assets); contributions[name]=rc
        if wealth["Insolvência"]:
            warnings.append(dbc.Alert(name+": perda de 100% ou mais em um período; evolução patrimonial interrompida na insolvência. Não simula chamadas de margem.",color="danger"))
        wealth_fig.add_scatter(x=[prices.index[0],*wealth['wealth'].index],y=[100.,*(wealth['wealth']*100)],name=name,line=dict(color=colors[name],width=2))
        drawdown_fig.add_scatter(x=[prices.index[0],*wealth['drawdown'].index],y=[0.,*wealth['drawdown']],name=name,line=dict(color=colors[name],width=2))
        runup_fig.add_scatter(x=[prices.index[0],*wealth['runup'].index],y=[0.,*wealth['runup']],name=name,line=dict(color=colors[name],width=2))
        pcr_fig.add_bar(x=assets,y=rc.PCR,name=name,marker_color=colors[name])
    if benchmark is not None and benchmark.reindex(returns.index).notna().all():
        bw=wealth_metrics(benchmark.reindex(returns.index))["wealth"]
        wealth_fig.add_scatter(x=[prices.index[0],*bw.index],y=[100.,*(bw*100)],name="BOVA11",line=dict(color="#B78127",dash="dash"))
    elif benchmark is not None:
        warnings.append(dbc.Alert("BOVA11 tem lacunas; beta usa datas em comum e a curva acumulada do benchmark não é exibida.",color="warning"))
    wealth_fig.update_layout(height=420,yaxis_title="Patrimônio (base 100)",hovermode="x unified")
    drawdown_fig.update_layout(height=420,yaxis_tickformat=".1%",yaxis_title="Drawdown",hovermode="x unified")
    runup_fig.update_layout(height=420,yaxis_tickformat=".1%",yaxis_title="Run-up",hovermode="x unified")
    pcr_fig.update_layout(height=420,barmode="group",yaxis_tickformat=".1%",yaxis_title="Contribuição percentual ao risco",xaxis_tickangle=-45)
    pcr_fig.update_yaxes(zeroline=True,zerolinecolor=GRAY)
    risk_table=pd.DataFrame(metrics).rename_axis("Indicador").reset_index()
    risk_export = risk_table.copy()
    risk_export.insert(0, "Frequência", freq)
    risk_export["Unidade"] = ["razão" if k in ("Beta vs. BOVA11", "Sharpe") else "decimal (0.01 = 1%)" for k in risk_export.Indicador]
    risk_csv="data:text/csv;base64,"+base64.b64encode(risk_export.to_csv(index=False).encode("utf-8-sig")).decode("ascii")
    for col in strategies:
        risk_table[col]=["—" if not np.isfinite(v) else (f"{v:.3f}" if k in ("Beta vs. BOVA11","Sharpe") else f"{v:.2%}") for k,v in zip(risk_table.Indicador,risk_table[col])]
    rc_table=pd.DataFrame({"Ativo":assets})
    for name,rc in contributions.items():
        for col in ("MCR","CR","PCR"):
            rc_table[f"{col} · {name}"]=rc[col].to_numpy()
    rc_formats={c:"{:.2%}" for c in rc_table.columns if c!="Ativo"}
    holdings=holdings.merge(rc_table,on="Ativo",how="left")
    holdings["Beta vs. BOVA11"]=[beta_against_market(returns[a],benchmark) if benchmark is not None else np.nan for a in holdings.Ativo]
    formats.update(rc_formats); formats["Beta vs. BOVA11"]="{:.3f}"
    horizon="1 mês" if freq=="Mensal" else "1 dia"
    risk_panels=[
        panel("Risk & Performance Analytics",html.Div([table(risk_table),html.A("Baixar métricas de risco (CSV)",href=risk_csv,download=f"risk_{freq.lower()}.csv",className="secondary-button")]),f"VaR: {horizon} · quantil de perda (pode ser negativo) · in-sample"),
        html.Div([panel("Evolução patrimonial",graph(wealth_fig),"Pesos constantes · in-sample"),
                  panel("Drawdown histórico",graph(drawdown_fig))],className="equal-grid"),
        html.Div([panel("Run-up histórico",graph(runup_fig),"Ganho em relação ao vale patrimonial anterior"),
                  panel("Contribuição percentual ao risco (PCR)",graph(pcr_fig),"Contribuições assinadas; total = 100% quando a volatilidade é positiva")],className="equal-grid"),
        panel("Decomposição do risco",table(rc_table,rc_formats),"MCR e CR: volatilidade anualizada; PCR: participação no risco")
    ]

    csv=holdings.to_csv(index=False).encode("utf-8-sig")
    href="data:text/csv;base64,"+base64.b64encode(csv).decode("ascii")
    matrix=cov_df.rename_axis("Ativo").reset_index()
    return html.Div([
        html.P(f"{len(assets)} ativos · {len(returns)} observações · fator de anualização {annualization_factor(freq)}",className="page-subtitle"),
        *warnings,
        html.Div([kpi_card("Retorno esperado",pct(tr),"Máximo Sharpe",tr>0),kpi_card("Volatilidade",pct(tv),"Anualizada"),kpi_card("Sharpe",f"{sharpe:.3f}" if np.isfinite(sharpe) else "—","Máximo Sharpe"),kpi_card("Taxa livre de risco",pct(rf),"Selic histórica anualizada")],className="kpi-grid"),
        html.Div([panel("Fronteira eficiente",graph(fig)),panel("Alocação ótima",graph(allocation),"Pesos comprados e vendidos")],className="chart-grid"),
        html.Div([panel("Mínima variância",html.P(f"Retorno: {pct(mr)} · Volatilidade: {pct(mv)}")),panel("Máximo Sharpe",html.P(f"Retorno: {pct(tr)} · Volatilidade: {pct(tv)}"))],className="equal-grid"),
        *risk_panels,
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

        html.A("Baixar pesos e indicadores (CSV)",href=href,download=f"markowitz_{freq.lower()}.csv",className="secondary-button")
    ])


@app.callback(Output("results","children"),Input("run","n_clicks"),State("tickers","value"),State("period","start_date"),State("period","end_date"),State("frequency","value"),State("max-weight","value"),State("points","value"),State("allow-short","value"),State("gross-limit","value"),State("net-target","value"),prevent_initial_call=True)


def run_optimization(n, tickers_text, start, end, frequency, max_weight, points, short_mode, gross_value, net_value):
    try:
        tickers=list(dict.fromkeys(re.findall(r"\b[A-Z]{4}\d{1,2}\b",(tickers_text or "").upper())))
        if len(tickers)<2 or len(tickers)>60:
            raise ValueError("Informe entre 2 e 60 tickers válidos.")
        start,end=date.fromisoformat(start[:10]),date.fromisoformat(end[:10])
        if start>=end or end>date.today():
            raise ValueError("Período inválido.")
        if frequency not in {"Mensal", "Diária", "Ambas"}:
            raise ValueError("Frequência inválida.")
        if not (5 <= max_weight <= 100 and 30 <= points <= 120):
            raise ValueError("Peso máximo ou número de pontos inválido.")
        limit=max_weight/100
        gross_limit=gross_value/100
        net_target=net_value/100
        allow_short=short_mode=="yes"
        if not 1 <= net_target <= gross_limit <= 3:
            raise ValueError("Exposição líquida deve estar entre 100% e a exposição bruta máxima.")
        if limit*len(tickers)<net_target-1e-9:
            raise ValueError("Limite por ativo incompatível com a exposição líquida.")
        freqs=["Mensal","Diária"] if frequency=="Ambas" else [frequency]
        tabs=[]
        for freq in freqs:
            try:
                children=render_results(tickers,start,end,freq,limit,points,allow_short,gross_limit,net_target)
            except Exception as exc:
                children=dbc.Alert(f"Erro na análise {freq}: {exc}",color="danger")
            tabs.append(dcc.Tab(label=freq,children=children))
        return html.Div([html.H2("Resultados da otimização",className="section-title"),dcc.Tabs(children=tabs)])
    except Exception as exc:
        return dbc.Alert(str(exc),color="danger")
if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",8050)))
