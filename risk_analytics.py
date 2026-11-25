"""Portfolio optimization and risk metrics; returns and weights are decimals."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, LinearConstraint
from scipy.stats import norm


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
            'Maximum Drawdown':float(dd.min()),'Maximum Run-up':float(ru.max()),
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
