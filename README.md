
# Markowitz Portfolio Analyzer

Dashboard de otimização de carteiras de ações brasileiras com Python + Streamlit.

## Funcionalidades da versão 1
- Universo inicial de ações associadas ao Ibovespa, editável pelo usuário.
- Inclusão de quaisquer tickers B3 aceitos pelo Yahoo Finance (sufixo `.SA` aplicado automaticamente).
- Importação opcional de CSV com coluna `ticker`.
- Frequência mensal e diária, inclusive análise simultânea.
- Retornos simples sobre preços ajustados.
- Retorno esperado pela média histórica e covariância amostral anualizada.
- Carteira de mínima variância (MVP), fronteira eficiente e carteira de máximo Sharpe.
- Restrição long-only (sem vendas a descoberto) e limite configurável por ativo.
- Selic histórica via Banco Central/SGS.
- Matrizes de correlação/covariância, pesos, gráficos e exportação CSV.

## Metodologia e convenções
- Mensal: retorno médio x 12; covariância x 12; Selic média mensal x 12. Isso replica a convenção da planilha de referência.
- Diária: retorno médio x 252; covariância x 252; Selic diária média capitalizada por 252 dias.
- Preços: Yahoo Finance com `auto_adjust=True`, isto é, série ajustada.
- Retornos: simples, `P_t / P_(t-1) - 1`.
- Otimização: SLSQP, pesos entre zero e o limite definido, soma dos pesos igual a 1.
- A fronteira eficiente é calculada para retornos-alvo a partir da carteira MVP até o maior retorno esperado individual.

## Importante sobre o universo Ibovespa
A lista pré-carregada é uma lista inicial editável de tickers associados ao universo brasileiro/Ibovespa, não uma composição oficial atualizada automaticamente. Para evitar afirmar uma composição histórica ou atual sem validação, atualize os tickers manualmente ou importe um CSV da carteira teórica vigente da B3. O modo personalizado aceita outros tickers negociados na B3, desde que disponíveis no provedor.

## Rodar localmente
Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar gratuitamente no Streamlit Community Cloud
1. Crie um repositório no GitHub e envie `app.py`, `requirements.txt` e este README.
2. Acesse https://share.streamlit.io e entre com sua conta GitHub.
3. Clique em **Create app**.
4. Escolha o repositório, a branch e `app.py` como arquivo principal.
5. Clique em **Deploy**.

A publicação requer autorização da conta GitHub pelo Streamlit. O código está pronto para essa etapa, mas o deploy não é realizado automaticamente por este pacote.

## Limitações conhecidas da versão 1
- Yahoo Finance/yfinance é uma fonte gratuita não oficial, sujeita a indisponibilidade, mudanças e limites.
- Tickers e disponibilidade histórica dependem do provedor.
- A lista inicial do Ibovespa não é atualizada automaticamente.
- A otimização média-variância é sensível às estimativas de retorno e covariância; não inclui custos de transação, impostos, liquidez ou restrições setoriais.
- A aplicação exige ao menos 20 observações completas por ativo selecionado.
- A fronteira e o máximo Sharpe são estimativas in-sample, não previsões de retorno futuro.
