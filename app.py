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
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
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
        if cash_returns is None: raise ValueError('Selic necessária para caixa ou financiamento residual.')
        rates=cash_returns.reindex(r.index)
        if rates.isna().any(): raise ValueError('Selic incompleta nas datas da carteira.')
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
# =========================================================
# INVESTOR PROFILE / STRATEGIC ASSET ALLOCATION
# =========================================================

PROFILE_A = {"Conservador": 12.0, "Moderado": 5.0, "Agressivo": 2.0}
PROFILE_COLORS = {"Conservador": "#497B98", "Moderado": "#B78127", "Agressivo": "#16836B"}

# Educational questionnaire: ordinal tolerance and separate capacity gates.
# None of these scores or limits is a validated suitability model.
TOLERANCE_QUESTIONS = [
    ("risk-preference", "1. Qual é sua preferência entre risco e retorno?", [
        "Preservar o capital", "Crescer com pouca oscilação", "Crescer aceitando oscilações relevantes",
        "Buscar maior retorno potencial, aceitando alta volatilidade"]),
    ("risk-reaction", "2. Se a carteira caísse 20% em três meses, como reagiria?", [
        "Resgataria tudo", "Reduziria a exposição", "Manteria a estratégia",
        "Avaliaria aumentar a exposição, considerando os fundamentos"]),
    ("risk-emotional", "3. Qual queda temporária conseguiria tolerar emocionalmente?", [
        "Até 5%", "Até 10%", "Até 20%", "Acima de 20%"]),
    ("risk-scenario", "4. Qual cenário hipotético preferiria?", [
        "Ganho pequeno, baixa incerteza", "Ganho moderado, pequenas perdas possíveis",
        "Ganho potencial elevado, perdas relevantes possíveis",
        "Ganho potencial muito elevado, perdas expressivas possíveis"]),
]
# Each alternative: label, value, ceiling on risky assets. These are transparent
# demonstration parameters, not empirically calibrated allocations or loss limits.
CAPACITY_QUESTIONS = [
    ("capacity-goal", "5. Objetivo principal deste patrimônio", [
        ("Reserva de emergência", "reserve", 0.0), ("Aquisição de bem ou despesa planejada", "purchase", .50),
        ("Geração de renda", "income", .60), ("Acumulação patrimonial", "growth", 1.0),
        ("Aposentadoria de longo prazo", "retire", 1.0)]),
    ("capacity-horizon", "6. Horizonte de investimento", [
        ("Menos de 1 ano", "short", 0.0), ("1 a 3 anos", "medium", .25),
        ("3 a 5 anos", "long", .60), ("Mais de 5 anos", "verylong", 1.0)]),
    ("capacity-withdrawal", "7. Quanto poderá precisar resgatar em 12 meses?", [
        ("Mais de 50%", "high", .25), ("25% a 50%", "mid", .50),
        ("Até 25%", "low", .75), ("Nenhum", "none", 1.0)]),
    ("capacity-reserve", "8. Reserva de emergência em meses de despesas", [
        ("Não possuo", "none", 0.0), ("Menos de 3 meses", "low", .25),
        ("3 a 6 meses", "mid", .60), ("Mais de 6 meses", "high", 1.0)]),
    ("capacity-income", "9. Estabilidade da renda", [
        ("Instável", "unstable", .25), ("Parcialmente previsível", "partial", .60),
        ("Estável", "stable", 1.0), ("Estável com capacidade de poupança", "high", 1.0)]),
    ("capacity-loss", "10. Perda suportável sem comprometer despesas essenciais ou objetivos", [
        ("Nenhuma", "none", 0.0), ("Até 5%", "5", .25), ("Até 10%", "10", .50),
        ("Até 20%", "20", .75), ("Acima de 20%", "more", 1.0)]),
]
KNOWLEDGE_QUESTIONS = [
    ("knowledge-experience", "11. Maior nível de experiência com investimentos", [
        ("Nenhuma experiência", "none"), ("Renda fixa e fundos", "fixed"),
        ("Ações e ETFs", "stocks"), ("Derivativos ou operações alavancadas", "advanced")]),
    ("knowledge-stock", "12. Uma ação pode perder mais de 30% em um ano?", [("Sim", "yes"), ("Não", "no"), ("Não sei", "unknown")]),
    ("knowledge-short", "13. Uma posição vendida pode gerar perdas superiores ao capital destinado à operação?", [("Sim", "yes"), ("Não", "no"), ("Não sei", "unknown")]),
    ("knowledge-leverage", "14. Alavancagem amplifica tanto ganhos quanto perdas?", [("Sim", "yes"), ("Não", "no"), ("Não sei", "unknown")]),
    ("knowledge-var", "15. VaR de 95% representa a maior perda possível?", [("Sim", "yes"), ("Não", "no"), ("Não sei", "unknown")]),
]
# Lotteries with two equiprobable annual returns; values are decimal rates.
CE_QUESTIONS = [
    ("ce-one", "16. Retorno de 0% ou 20%, com 50% de chance cada", 0., .20),
    ("ce-two", "17. Retorno de 2% ou 18%, com 50% de chance cada", .02, .18),
    ("ce-three", "18. Retorno de -4% ou 16%, com 50% de chance cada", -.04, .16),
]


def classify_investor(answers):
    if len(answers) != len(TOLERANCE_QUESTIONS) or any(a not in (0,1,2,3) for a in answers):
        raise ValueError("Responda às quatro perguntas de tolerância.")
    score = int(sum(answers))
    name = "Conservador" if score <= 4 else ("Moderado" if score <= 8 else "Agressivo")
    return {"profile": name, "score": score, "A": PROFILE_A[name]}


def evaluate_profile(tolerance, capacity, knowledge, calibration="score", ce_answers=None):
    result = classify_investor(tolerance)
    if len(capacity) != len(CAPACITY_QUESTIONS) or len(knowledge) != len(KNOWLEDGE_QUESTIONS):
        raise ValueError("Preencha todo o questionário.")
    limits = []
    for (_, question, opts), answer in zip(CAPACITY_QUESTIONS, capacity):
        matches = [(label, limit) for label, val, limit in opts if val == answer]
        if not matches:
            raise ValueError("Responda todas as perguntas de capacidade financeira.")
        label, limit = matches[0]
        limits.append((question, label, limit))
    for (_, question, opts), answer in zip(KNOWLEDGE_QUESTIONS, knowledge):
        if answer not in {value for _, value in opts}:
            raise ValueError("Responda todas as perguntas de conhecimento.")
    correct = ("yes", "yes", "yes", "no")
    knowledge_score = sum(a == b for a, b in zip(knowledge[1:], correct))
    # No automatic access to complex strategies, regardless of quiz score.
    knowledge_cap = 1.0
    if knowledge[0] == "none" or knowledge[1] != "yes":
        knowledge_cap = .25
    elif knowledge[0] == "fixed" or knowledge[4] != "no":
        knowledge_cap = .60
    limits.append(("Experiência e compreensão de risco básico", f"{knowledge_score}/4 respostas corretas", knowledge_cap))
    risk_cap = min(item[2] for item in limits)
    result.update({"cap": risk_cap, "limits": limits, "knowledge_score": knowledge_score,
                   "calibration": calibration, "ce_values": []})
    if calibration == "ce":
        if ce_answers is None or len(ce_answers) != 3 or any(x is None for x in ce_answers):
            raise ValueError("Responda aos três cenários de equivalente de certeza.")
        aversions = []
        for (_, _, low, high), ce in zip(CE_QUESTIONS, ce_answers):
            mean = (low+high)/2
            variance = ((high-low)/2)**2
            a = 2*(mean-float(ce))/variance
            if not np.isfinite(a) or not 2-1e-6 <= a <= 12+1e-6:
                raise ValueError("Equivalente de certeza inválido.")
            aversions.append(float(np.clip(a,2,12)))
        result["A"] = float(np.median(aversions))
        result["ce_values"] = aversions
    elif calibration != "score":
        raise ValueError("Método de calibração inválido.")
    return result


def personalized_portfolio(weights_tangent, mu, cov, rf, A, max_risky_allocation=1.,
                           max_volatility=None):
    """CAL allocation (long-only, no borrowing); inputs annualized, decimals."""
    w_t, mu, cov = np.asarray(weights_tangent,float), np.asarray(mu,float), np.asarray(cov,float)
    if not np.isfinite(A) or A <= 0 or not 0 <= max_risky_allocation <= 1:
        raise ValueError("Aversão ou limite inválido.")
    if (not np.isclose(w_t.sum(),1.,atol=2e-6) or min(w_t) < -2e-6):
        raise ValueError("A referência CAL deve ser long-only e 100% investida.")
    w_t = np.maximum(w_t,0)
    w_t /= w_t.sum()
    mu_t = float(w_t@mu)
    var_t = float(w_t@cov@w_t)
    if var_t <= 1e-12:
        raise ValueError("A carteira de referência tem variância praticamente nula; CAL indefinida.")
    ceiling = max_risky_allocation
    if max_volatility is not None:
        ceiling = min(ceiling, max_volatility / np.sqrt(var_t))
    theoretical = (mu_t-rf)/(A*var_t)
    y = float(np.clip(theoretical,0.,ceiling))
    return allocation_summary(y*w_t,mu,cov,rf,A, theoretical=theoretical, cal_ceiling=ceiling)


def allocation_summary(w,mu,cov,rf,A,theoretical=None,cal_ceiling=None):
    w = np.asarray(w,float)
    er,vol = portfolio_moments(w,mu,cov,rf)
    return {"A":float(A), "weights":w, "y":float(w.sum()), "weight_rf":float(1-w.sum()),
            "expected_return":er,"volatility":vol,"sharpe":(er-rf)/vol if vol>1e-10 else np.nan,
            "utility":float(er-A/2*vol**2),"y_theoretical":theoretical,"cal_ceiling":cal_ceiling}


def direct_profile_portfolio(mu,cov,rf,A,cap,limit,max_volatility=None):
    """Convex mean-variance utility, cash residual, no short or leverage."""
    mu,cov = np.asarray(mu,float),np.asarray(cov,float)
    if A <= 0 or not 0 <= cap <= 1 or limit <= 0:
        raise ValueError("Restrições de utilidade inválidas.")
    n = len(mu)
    if cap <= 1e-12:
        return allocation_summary(np.zeros(n),mu,cov,rf,A)
    ex = mu-rf
    scale = max(float(np.max(np.abs(ex))),A*float(np.max(np.diag(cov))),1e-4)
    constraints = [LinearConstraint(np.ones((1,n)),0.,cap)]
    if max_volatility is not None:
        vscale = max(max_volatility**2,float(np.max(np.diag(cov))),1e-8)
        constraints.append({"type":"ineq","fun":lambda w:(max_volatility**2-w@cov@w)/vscale,
                            "jac":lambda w:-2*cov@w/vscale})
    res = minimize(lambda w:(A/2*w@cov@w-ex@w)/scale,np.zeros(n),
                   jac=lambda w:(A*cov@w-ex)/scale,bounds=[(0.,min(limit,cap))]*n,
                   constraints=constraints,method="SLSQP",options={"maxiter":1000,"ftol":1e-11})
    w = np.maximum(res.x,0.)
    if not res.success or w.sum()>cap+1e-6 or max(w)>limit+1e-6 or (
            max_volatility is not None and w@cov@w>max_volatility**2+1e-8):
        raise RuntimeError("Otimização de utilidade não convergiu: "+str(res.message))
    return allocation_summary(w,mu,cov,rf,A)


def complete_frontier(mu,cov,rf,cap,limit,max_volatility=None,points=36):
    """Efficient frontier including cash with the same portfolio constraints."""
    mu,cov=np.asarray(mu,float),np.asarray(cov,float)
    n=len(mu); ex=mu-rf
    rows=[{"Retorno":rf,"Volatilidade":0.}]
    if cap<=1e-10 or ex.max()<=1e-10:
        return pd.DataFrame(rows)
    bounds=[(0.,min(cap,limit))]*n
    linear=LinearConstraint(np.ones((1,n)),0.,cap)
    varscale=max(float(np.max(np.diag(cov))),1e-8)
    ret_scale=max(float(np.max(np.abs(ex))),1e-6)
    lp=linprog(-ex,A_ub=np.ones((1,n)),b_ub=[cap],bounds=bounds,method="highs")
    if not lp.success:
        raise RuntimeError("Não foi possível encontrar o retorno máximo factível.")
    wmax=lp.x
    if max_volatility is not None and wmax@cov@wmax>max_volatility**2:
        seed=wmax*min(1.,max_volatility/np.sqrt(wmax@cov@wmax))
        con={"type":"ineq","fun":lambda w:(max_volatility**2-w@cov@w)/varscale,
             "jac":lambda w:-2*cov@w/varscale}
        res=minimize(lambda w:-ex@w/ret_scale,seed,jac=lambda w:-ex/ret_scale,
                     bounds=bounds,constraints=[linear,con],method="SLSQP",
                     options={"maxiter":1000,"ftol":1e-11})
        if not res.success or res.x@cov@res.x>max_volatility**2+1e-8 or res.x.sum()>cap+1e-6:
            raise RuntimeError("Não foi possível calcular o extremo da fronteira com limite de volatilidade.")
        wmax=res.x
    peak=float(ex@wmax)
    if peak<=1e-10: return pd.DataFrame(rows)
    for t in np.linspace(0,peak,max(2,points))[1:]:
        con=LinearConstraint((ex/ret_scale)[None,:],t/ret_scale,t/ret_scale)
        res=minimize(lambda w:w@cov@w/varscale,(t/peak)*wmax,
                     jac=lambda w:2*cov@w/varscale,bounds=bounds,
                     constraints=[linear,con],method="SLSQP",options={"maxiter":700,"ftol":1e-10})
        if not res.success: continue
        w=res.x
        vol=float(np.sqrt(max(w@cov@w,0.)))
        if abs(ex@w-t)>1e-6 or w.sum()>cap+1e-6 or (max_volatility is not None and vol>max_volatility+1e-6):
            continue
        rows.append({"Retorno":float(rf+ex@w),"Volatilidade":vol})
    return pd.DataFrame(rows)


def profile_questionnaire():
    def drop(qid, label, pairs):
        return field(label, dcc.Dropdown(id=qid,options=[{"label":l,"value":v} for l,v in pairs],
                                         placeholder="Selecione uma resposta",clearable=True))
    def block(title,subtitle,children):
        return html.Div([html.H3(title,className="chart-title"),html.P(subtitle,className="form-text"),
                         html.Div(children,style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(min(100%, 250px),1fr))","gap":"16px"})],
                        className="chart-card")
    ce_fields=[]
    for qid,q,low,high in CE_QUESTIONS:
        mean=(low+high)/2; var=((high-low)/2)**2
        choices=[(f"{(mean-a/2*var):.2%}",round(mean-a/2*var,12)) for a in (12,8,5,4,2)]
        ce_fields.append(drop(qid,q+" — retorno certo equivalente",choices))
    return html.Section([
        html.H2("Investor Profile & Strategic Asset Allocation",className="section-title"),
        html.P("Questionário educacional de tolerância, capacidade financeira e conhecimento. Os limites são regras ilustrativas, "
               "não suitability validado nem recomendação de investimento.",className="page-subtitle"),
        dbc.Alert("Calcule as carteiras na seção Parâmetros antes de gerar o relatório. O perfil utiliza a mesma amostra histórica; "
                  "por segurança metodológica, esta seção não permite vendas a descoberto nem financiamento.",color="info"),
        block("01 · Tolerância comportamental","Quatro respostas, pontuação de 0 a 12. Nenhuma alternativa é considerada melhor.",
              [drop(qid,q,[(label,i) for i,label in enumerate(opts)]) for qid,q,opts in TOLERANCE_QUESTIONS]),
        block("02 · Capacidade financeira e objetivos","As respostas impõem limites independentes; prevalece o menor teto de exposição.",
              [drop(qid,q,[(l,v) for l,v,_ in opts]) for qid,q,opts in CAPACITY_QUESTIONS]),
        block("03 · Conhecimento e experiência","As respostas de conhecimento são avaliadas separadamente do apetite a risco.",
              [drop(qid,q,opts) for qid,q,opts in KNOWLEDGE_QUESTIONS]),
        html.Details([
            html.Summary("Calibração opcional por equivalente de certeza"),
            html.P("Para cada loteria hipotética de dois resultados igualmente prováveis, indique o retorno certo que seria equivalente. "
                   "O modelo usa a mediana de três valores implícitos de A (intervalo ilustrativo de 2 a 12).",className="form-text"),
            html.Div(ce_fields,style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(min(100%, 240px),1fr))","gap":"16px"})],
            className="chart-card"),
        block("04 · Configuração do modelo","Selecione a frequência da amostra calculada e o método. Volatilidade-limite é uma estimativa, não garantia.",
              [field("Frequência",dcc.Dropdown(id="profile-frequency",options=[],value=None,clearable=False,disabled=True)),
               field("Método de alocação",dcc.Dropdown(id="profile-model",options=[
                   {"label":"CAL — tangente + Selic","value":"cal"},{"label":"Utilidade com restrições (otimização direta)","value":"utility"}],value="cal",clearable=False)),
               field("Calibração do coeficiente A",dcc.Dropdown(id="profile-calibration",options=[
                   {"label":"Pontuação comportamental (A = 12, 5 ou 2)","value":"score"},
                   {"label":"Equivalente de certeza (3 cenários)","value":"ce"}],value="score",clearable=False)),
               field("Teto adicional de volatilidade anual (opcional)",dcc.Dropdown(id="profile-volatility",options=[
                   {"label":"Sem teto adicional","value":0}]+[{"label":f"{v}% a.a.","value":v} for v in (5,10,15,20,25,30,40)],value=0,clearable=False))]),
        html.Button("Calcular carteira do perfil",id="profile-run",n_clicks=0,className="primary-button"),
        dcc.Loading(html.Div(id="profile-result",children=dbc.Alert("Aguardando cálculo e respostas.",color="info")),type="circle",color=GREEN),
    ],id="investor-profile",style={"marginTop":"24px","marginBottom":"24px","scrollMarginTop":"20px"})


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
        html.A("Perfil do investidor", href="#investor-profile", className="nav-link"),
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
        dcc.Store(id="profile-market",storage_type="memory"),
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
                        profile_questionnaire(),
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
                                html.H3("Perfil de risco e alocação estratégica"),
                                html.P("A pontuação comportamental de 0 a 12 define A=12, 5 ou 2; há uma calibração alternativa por equivalente de certeza "
                                       "de três loterias hipotéticas. A capacidade financeira e a experiência limitam a exposição separadamente. "
                                       "Os limites são heurísticos e não constituem suitability validado."),
                                html.P("A CAL usa y* = (retorno esperado da tangente - Selic)/(A × variância da tangente), com y entre 0 e o teto admissível. "
                                       "A otimização direta maximiza U = retorno esperado − A/2 × variância, incluindo caixa residual, limite individual, "
                                       "teto de exposição e limite opcional de volatilidade. O módulo não permite short nem alavancagem; "
                                       "essas estratégias continuam disponíveis na análise exploratória. A Selic é proxy teórica, não ativo sem risco real."),
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



def render_results(tickers, start, end, freq, limit, points, allow_short=False, gross_limit=1., net_target=1., include_snapshot=False):
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
    result_view = html.Div([
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


    if include_snapshot:
        return result_view, market_snapshot(freq,assets,prices,returns,mu,cov,rf,mvp,tan,frontier,
                                             selic,benchmark,limit,allow_short,net_target)
    return result_view


def market_snapshot(freq,assets,prices,returns,mu,cov,rf,mvp,tan,frontier,selic,benchmark,limit,allow_short,net_target):
    """Browser-memory cache of market data only; no answers or identity."""
    front=frontier[["Retorno","Volatilidade"]].to_dict("records")
    return {"freq":freq,"assets":list(assets),"index":[x.isoformat() for x in returns.index],
            "initial_date":prices.index[0].isoformat(),"returns":returns.to_numpy().tolist(),
            "mu":mu.tolist(),"cov":cov.tolist(),"rf":float(rf),"mvp":mvp.tolist(),"tan":tan.tolist(),
            "frontier":front,"selic":selic.reindex(returns.index).to_numpy().tolist(),
            "benchmark":[None if pd.isna(x) else float(x) for x in benchmark.reindex(returns.index)] if benchmark is not None else None,
            "limit":float(limit),"reference_ok":bool(not allow_short and abs(net_target-1)<1e-9)}


def unpack_snapshot(data):
    assets=list(data["assets"])
    index=pd.to_datetime(data["index"])
    returns=pd.DataFrame(data["returns"],index=index,columns=assets,dtype=float)
    selic=pd.Series(data["selic"],index=index,dtype=float)
    bench=pd.Series(data["benchmark"],index=index,dtype=float) if data["benchmark"] is not None else None
    mu=np.asarray(data["mu"],float); cov=np.asarray(data["cov"],float); rf=float(data["rf"])
    if len(returns)<20 or len(assets)<2 or not np.isfinite(returns.to_numpy()).all() or not np.isfinite(selic).all():
        raise ValueError("Amostra de mercado inválida ou incompleta.")
    if mu.shape!=(len(assets),) or cov.shape!=(len(assets),len(assets)) or not np.isfinite(cov).all() or not np.isfinite(mu).all() or not np.isfinite(rf):
        raise ValueError("Estatísticas de mercado inválidas.")
    if np.linalg.eigvalsh(cov).min() < -1e-8 or not np.allclose(cov,cov.T):
        raise ValueError("Matriz de covariância inválida.")
    return assets,returns,selic,bench,mu,cov,rf


def profile_metrics(w,mu,cov,rf,returns,selic,benchmark,A):
    moments=allocation_summary(w,mu,cov,rf,A)
    rp=portfolio_return_series(returns,w,selic)
    perf=wealth_metrics(rp)
    out={"Retorno esperado anualizado":moments["expected_return"],
         "Volatilidade anualizada (modelo)":moments["volatility"],
         "Sharpe (modelo)":moments["sharpe"],
         "Equivalente de certeza / utilidade":moments["utility"],
         "Beta vs. BOVA11":beta_against_market(rp,benchmark) if benchmark is not None else np.nan}
    out.update(var_metrics(rp))
    out.update({key:perf[key] for key in ("Retorno acumulado","Maximum Drawdown","Maximum Run-up")})
    out.update(exposure(w))
    return out,perf,rp


def profile_report(data,profile,method,max_volatility=None):
    assets,returns,selic,benchmark,mu,cov,rf=unpack_snapshot(data)
    cap=profile["cap"]; A=profile["A"]; limit=float(data["limit"])
    freq=data["freq"]
    # Retain the original risky reference only if the exploratory portfolio
    # was itself long-only and unleveraged. Otherwise recompute on cached data.
    if data["reference_ok"]:
        mvp=np.asarray(data["mvp"],float); tan=np.asarray(data["tan"],float)
        frontier=pd.DataFrame(data["frontier"])
    else:
        mvp,tan,frontier=optimize_portfolios(mu,cov,rf,points=45,limit=limit,
                                           allow_short=False,gross_limit=1.,net_target=1.)
    tr,tv=portfolio_moments(tan,mu,cov,rf)
    if method not in ("cal","utility"):
        raise ValueError("Método de alocação inválido.")
    def choose(a):
        if method=="cal":
            return personalized_portfolio(tan,mu,cov,rf,a,cap,max_volatility)
        return direct_profile_portfolio(mu,cov,rf,a,cap,limit,max_volatility)
    selected=choose(A)
    examples={name:choose(a) for name,a in PROFILE_A.items()}
    selected_name="Carteira do perfil"
    w=selected["weights"]
    # Correctly distinguish a risky-only frontier, CAL, and the full constrained
    # frontier. The CAL is not extended into leverage in this module.
    fig=go.Figure()
    fig.add_scatter(x=frontier.Volatilidade,y=frontier.Retorno,mode="lines+markers",
                    name="Fronteira long-only (100% em risco)",line=dict(color="#94A3B8",width=2),marker=dict(size=3))
    fig.add_scatter(x=[0,tv],y=[rf,tr],mode="lines",name="CAL de referência",
                    line=dict(color=NAVY,dash="dash",width=2))
    fig.add_scatter(x=[0],y=[rf],mode="markers",name="Selic",marker=dict(color=GRAY,size=10))
    mr,mv=portfolio_moments(mvp,mu,cov,rf)
    fig.add_scatter(x=[mv],y=[mr],mode="markers",name="MVP de referência",marker=dict(color="#497B98",symbol="diamond",size=10))
    fig.add_scatter(x=[tv],y=[tr],mode="markers",name="Tangente de referência",marker=dict(color="#B78127",size=11,symbol="star"))
    if method=="cal":
        bound=selected["cal_ceiling"]
        fig.add_scatter(x=[0,bound*tv],y=[rf,rf+bound*(tr-rf)],mode="lines",
                        name="CAL factível no perfil",line=dict(color=GREEN,width=3))
    else:
        full=complete_frontier(mu,cov,rf,cap,limit,max_volatility,36)
        fig.add_scatter(x=full.Volatilidade,y=full.Retorno,mode="lines+markers",
                        name="Fronteira completa restrita",line=dict(color=GREEN,width=3),marker=dict(size=3))
    # Group coincident points so hover/labels remain accessible.
    groups={}
    for name,p in examples.items():
        key=(round(p["volatility"],7),round(p["expected_return"],7))
        groups.setdefault(key,[]).append(name)
    for (risk,ret),names in groups.items():
        label=" / ".join(names)
        fig.add_scatter(x=[risk],y=[ret],mode="markers",name=label,
                        marker=dict(color=PROFILE_COLORS[names[0]],size=10,symbol="circle-open",line=dict(width=2)),
                        text=[label],hovertemplate="%{text}<br>Volatilidade %{x:.2%}<br>Retorno %{y:.2%}<extra></extra>")
    sx=selected["volatility"]; sy=selected["expected_return"]
    fig.add_scatter(x=[sx],y=[sy],mode="markers",name="Perfil selecionado",
                    marker=dict(size=17,symbol="star",color=GREEN,line=dict(color=WHITE,width=1.5)),
                    hovertemplate="Carteira do perfil<br>Volatilidade %{x:.2%}<br>Retorno %{y:.2%}<extra></extra>")
    # Iso-utility curve through the solution, on the feasible plotted risk range.
    xs=np.linspace(0,max(tv,sx,float(frontier.Volatilidade.max()),.001)*1.05,120)
    fig.add_scatter(x=xs,y=selected["utility"]+A/2*xs**2,mode="lines",name="Curva de indiferença",
                    line=dict(color="#B78127",width=1.7,dash="dot"),visible="legendonly")
    fig.update_layout(height=520,hovermode="closest",xaxis_title="Volatilidade anualizada",
                      yaxis_title="Retorno esperado anualizado",xaxis_tickformat=".1%",yaxis_tickformat=".1%",
                      legend=dict(orientation="h",y=-.22,yanchor="top",x=0,xanchor="left"))
    # Readable full-weight chart for all hypothetical profiles, same capacity gates.
    names=list(examples)+[selected_name]
    allocations={**examples,selected_name:selected}
    alloc_fig=go.Figure()
    allocation_data=pd.DataFrame({"Ativo":[*assets,"Selic (caixa)"]})
    for name,p in allocations.items():
        allocation_data[name]=[*p["weights"],p["weight_rf"]]
    for i,asset in enumerate(allocation_data.Ativo):
        if max(abs(allocation_data.loc[i,n]) for n in names)<1e-8: continue
        alloc_fig.add_bar(x=names,y=[allocation_data.loc[i,n] for n in names],name=asset,
                          marker_color=NAVY if i==len(assets) else COLOR_PALETTE[(i+1)%len(COLOR_PALETTE)],
                          hovertemplate="%{x}<br>"+asset+": %{y:.2%}<extra></extra>")
    alloc_fig.update_layout(height=450,barmode="stack",yaxis_tickformat=".0%",yaxis_title="Peso no patrimônio",legend=dict(orientation="h",y=-.2,yanchor="top"))
    profile_compare=[]
    for name,p in allocations.items():
        profile_compare.append({"Perfil":name,"A":f"{p['A']:.2f}","Exposição ao risco":pct(p["y"]),
                                "Selic":pct(p["weight_rf"]),"Retorno":pct(p["expected_return"]),
                                "Volatilidade":pct(p["volatility"]),
                                "Sharpe":f"{p['sharpe']:.3f}" if np.isfinite(p["sharpe"]) else "—"})
    # Reference portfolios and selected portfolio use same dates and same A.
    strategies={"MVP (referência)":mvp,"Tangente (referência)":tan,selected_name:w}
    perf_fig=go.Figure(); dd_fig=go.Figure(); ru_fig=go.Figure()
    metrics={}; rc_frames={}
    colors={"MVP (referência)":"#497B98","Tangente (referência)":"#B78127",selected_name:GREEN}
    warnings=[]
    initial=pd.Timestamp(data["initial_date"])
    for name,weights in strategies.items():
        metric,perf,rp=profile_metrics(weights,mu,cov,rf,returns,selic,benchmark,A)
        metric["Volatilidade histórica anualizada"]=float(rp.std(ddof=1)*np.sqrt(annualization_factor(freq)))
        metrics[name]=metric
        rc_frames[name]=risk_contribution(weights,cov,assets)
        perf_fig.add_scatter(x=[initial,*perf["wealth"].index],y=[100.,*(perf["wealth"]*100)],name=name,line=dict(color=colors[name],width=2))
        dd_fig.add_scatter(x=[initial,*perf["drawdown"].index],y=[0.,*perf["drawdown"]],name=name,line=dict(color=colors[name],width=2))
        ru_fig.add_scatter(x=[initial,*perf["runup"].index],y=[0.,*perf["runup"]],name=name,line=dict(color=colors[name],width=2))
        if perf["Insolvência"]: warnings.append(dbc.Alert(name+": trajetória interrompida por insolvência.",color="danger"))
    for bench_name,series,color in (("BOVA11",benchmark,GRAY),("Selic",selic,NAVY)):
        if series is not None and series.notna().all():
            p=wealth_metrics(series)
            perf_fig.add_scatter(x=[initial,*p["wealth"].index],y=[100.,*(p["wealth"]*100)],name=bench_name,line=dict(color=color,dash="dash"))
    if benchmark is None or benchmark.dropna().size<20:
        warnings.append(dbc.Alert("BOVA11 indisponível ou insuficiente: beta não estimado.",color="warning"))
    elif benchmark.isna().any():
        warnings.append(dbc.Alert("BOVA11 possui lacunas: beta usa datas em comum, curva patrimonial omitida.",color="warning"))
    if len(returns)<500:
        warnings.append(dbc.Alert("VaR histórico de 99% pouco preciso: menos de 500 observações.",color="warning"))
    if cap<=1e-10:
        warnings.append(dbc.Alert("Uma restrição de capacidade impede exposição a ativos arriscados neste cenário. "
                                  "O resultado ilustra 100% na proxy Selic; não equivale a uma recomendação de produto.",color="warning"))
    if method=="cal" and selected["y_theoretical"]>selected["cal_ceiling"]+1e-8:
        warnings.append(dbc.Alert(f"Alocação teórica de {pct(selected['y_theoretical'])} limitada a {pct(selected['cal_ceiling'])}. "
                                  "Não há financiamento ou alavancagem no módulo de perfil.",color="info"))
    if profile["ce_values"]:
        warnings.append(dbc.Alert("A escolhido pela mediana dos cenários: "+", ".join(f"{a:.2f}" for a in profile["ce_values"])+
                                  ". A categoria comportamental continua baseada nas quatro respostas de tolerância.",color="info"))
    mg=pd.DataFrame(metrics).rename_axis("Indicador").reset_index()
    raw_metrics=mg.copy()
    for col in strategies:
        mg[col]=["—" if not np.isfinite(v) else (f"{v:.3f}" if k in ("Beta vs. BOVA11","Sharpe (modelo)") else pct(v))
                 for k,v in zip(mg.Indicador,mg[col])]
    pcr=rc_frames[selected_name].copy()
    pcr.insert(1,"Peso",w)
    # Selic is modelled as deterministic here, not included in risky covariance.
    cash_row=pd.DataFrame({"Ativo":["Selic (caixa)"],"Peso":[selected["weight_rf"]],
                          "MCR":[np.nan],"CR":[0. if sx>1e-10 else np.nan],"PCR":[0. if sx>1e-10 else np.nan]})
    pcr_display=pd.concat([pcr,cash_row],ignore_index=True)
    pcr_fig=go.Figure()
    for name,rc in rc_frames.items():
        pcr_fig.add_bar(x=assets,y=rc.PCR,name=name,marker_color=colors[name])
    pcr_fig.update_layout(height=420,barmode="group",yaxis_tickformat=".1%",yaxis_title="PCR (risco do modelo)")
    if sx<=1e-10:
        pcr_fig.add_annotation(text="PCR do perfil indefinido: variância de risco do modelo nula.",xref="paper",yref="paper",x=.5,y=1.07,showarrow=False)
    perf_fig.update_layout(height=440,yaxis_title="Patrimônio (base 100)",hovermode="x unified")
    dd_fig.update_layout(height=380,yaxis_title="Drawdown",yaxis_tickformat=".1%",hovermode="x unified")
    ru_fig.update_layout(height=380,yaxis_title="Run-up",yaxis_tickformat=".1%",hovermode="x unified")
    cap_table=pd.DataFrame([{"Critério":q,"Resposta":answer,"Teto ilustrativo":pct(lim),
                            "Restritivo":"Sim" if abs(lim-cap)<1e-8 else "Não"} for q,answer,lim in profile["limits"]])
    methodology="CAL: combinação da referência long-only e Selic." if method=="cal" else "Utilidade direta: pesos individuais, caixa residual, limite de exposição e teto opcional de volatilidade."
    raw_metrics.insert(0,"Frequência",freq)
    raw_metrics["Unidade"]=["razão" if k in ("Beta vs. BOVA11","Sharpe (modelo)") else "decimal (0.01 = 1%)" for k in raw_metrics.Indicador]
    def csv_link(frame,name,label):
        href="data:text/csv;base64,"+base64.b64encode(frame.to_csv(index=False).encode("utf-8-sig")).decode("ascii")
        return html.A(label,href=href,download=name,className="secondary-button",style={"display":"inline-block","margin":"8px 10px 0 0"})
    subtitle=f"{freq} · {len(returns)} observações · {initial:%d/%m/%Y} a {returns.index[-1]:%d/%m/%Y}"
    return html.Div([
        html.H3("Investor Risk Report",className="section-title",style={"marginTop":"20px"}),
        html.P(subtitle+" · "+methodology,className="page-subtitle"),
        *warnings,
        html.Div([kpi_card("Perfil comportamental",profile["profile"],f"Pontuação {profile['score']}/12"),
                  kpi_card("Aversão ao risco",f"{A:.2f}","Coeficiente A"),
                  kpi_card("Teto de risco",pct(cap),"Capacidade + conhecimento (heurística)"),
                  kpi_card("Conhecimento",f"{profile['knowledge_score']}/4","Questões objetivas")],className="kpi-grid"),
        html.Div([kpi_card("Retorno esperado",pct(sy),"Anualizado",sy>0),
                  kpi_card("Volatilidade",pct(sx),"Modelo anualizado"),
                  kpi_card("Sharpe",f"{selected['sharpe']:.3f}" if np.isfinite(selected["sharpe"]) else "—","Modelo"),
                  kpi_card("Utilidade",pct(selected["utility"]),"Equivalente de certeza")],className="kpi-grid"),
        html.Div([panel("Fronteira eficiente, CAL e carteira do perfil",graph(fig),
                        "Curva de indiferença disponível na legenda. Pontos coincidentes agrupados."),
                  panel("Alocação estratégica por perfil",graph(alloc_fig),
                        "Perfis de referência A=12, 5 e 2 sob as mesmas restrições de capacidade")],className="chart-grid"),
        panel("Comparação dos perfis",table(pd.DataFrame(profile_compare))),
        panel("Pesos por ativo — inclui Selic",table(allocation_data,{n:"{:.2%}" for n in names})),
        panel("Estatísticas de risco e performance",html.Div([table(mg),csv_link(raw_metrics,f"perfil_risco_{freq.lower()}.csv","Baixar métricas (CSV)")]),
              "MVP e tangente são referências arriscadas; podem superar o teto do perfil. VaR no horizonte de "+("1 mês" if freq=="Mensal" else "1 dia")),
        panel("Evolução patrimonial",graph(perf_fig),"In-sample · pesos constantes rebalanceados a cada observação · base 100"),
        html.Div([panel("Maximum Drawdown",graph(dd_fig)),panel("Maximum Run-up",graph(ru_fig))],className="equal-grid"),
        panel("Contribuição percentual ao risco (PCR)",graph(pcr_fig),"Modelo de covariância dos ativos arriscados; Selic determinística."),
        panel("Decomposição de risco da carteira do perfil",html.Div([table(pcr_display,{c:"{:.2%}" for c in ("Peso","MCR","CR","PCR")}),
              csv_link(pcr_display,f"perfil_pcr_{freq.lower()}.csv","Baixar pesos e PCR (CSV)")])),
        panel("Regras de capacidade aplicadas",table(cap_table),"O menor teto é utilizado. Esses limites não garantem que a perda máxima fique abaixo da tolerância declarada."),
        html.P("Simulação acadêmica, não recomendação. A Selic é uma proxy teórica de caixa; taxas, impostos, liquidez e risco de produtos reais não são modelados. "
               "Os parâmetros são estimados na mesma amostra; desempenho não é backtest. A classificação, os valores de A e os limites de capacidade são ilustrativos.",className="form-text")
    ])


@app.callback(Output("results","children"),Output("profile-market","data"),Input("run","n_clicks"),State("tickers","value"),State("period","start_date"),State("period","end_date"),State("frequency","value"),State("max-weight","value"),State("points","value"),State("allow-short","value"),State("gross-limit","value"),State("net-target","value"),prevent_initial_call=True)


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
        snapshots={}
        for freq in freqs:
            try:
                children,snapshot=render_results(tickers,start,end,freq,limit,points,allow_short,gross_limit,net_target,include_snapshot=True)
                snapshots[freq]=snapshot
            except Exception as exc:
                children=dbc.Alert(f"Erro na análise {freq}: {exc}",color="danger")
            tabs.append(dcc.Tab(label=freq,children=children))
        return html.Div([html.H2("Resultados da otimização",className="section-title"),dcc.Tabs(children=tabs)]), snapshots
    except Exception as exc:
        return dbc.Alert(str(exc),color="danger"), {}


# ---- Profile callbacks ----
@app.callback(
    Output("profile-frequency","options"),Output("profile-frequency","value"),Output("profile-frequency","disabled"),
    Input("profile-market","data"),State("profile-frequency","value")
)
def available_profile_frequencies(data,current):
    freqs=[f for f in ("Mensal","Diária") if f in (data or {})]
    return ([{"label":f,"value":f} for f in freqs],
            current if current in freqs else (freqs[0] if freqs else None),not bool(freqs))


PROFILE_IDS=[q[0] for q in TOLERANCE_QUESTIONS+CAPACITY_QUESTIONS+KNOWLEDGE_QUESTIONS+CE_QUESTIONS]

@app.callback(
    Output("profile-result","children"),
    Input("profile-run","n_clicks"),Input("profile-market","data"),Input("profile-frequency","value"),
    Input("profile-model","value"),Input("profile-calibration","value"),Input("profile-volatility","value"),
    *[Input(qid,"value") for qid in PROFILE_IDS],
    prevent_initial_call=True
)
def run_profile(n,data,freq,method,calibration,volatility,*answers):
    # Do not recalculate on every answer; invalidate previous report on input changes.
    if ctx.triggered_id != "profile-run":
        return dbc.Alert("Parâmetros ou respostas atualizados. Clique em Calcular carteira do perfil para gerar o novo relatório.",color="info")
    if not data or freq not in data:
        return dbc.Alert("Calcule primeiro as carteiras em Parâmetros e selecione uma frequência disponível.",color="warning")
    try:
        if volatility not in (0,5,10,15,20,25,30,40):
            raise ValueError("Teto de volatilidade inválido.")
        profile=evaluate_profile(answers[:4],answers[4:10],answers[10:15],calibration,answers[15:18])
        return profile_report(data[freq],profile,method,volatility/100 if volatility else None)
    except Exception as exc:
        return dbc.Alert("Análise de perfil: "+str(exc),color="warning")


if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",8050)))
