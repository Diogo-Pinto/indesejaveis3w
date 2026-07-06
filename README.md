# Detecção de Eventos Indesejáveis em Poços de Petróleo Offshore — 3W Dataset (Petrobras)

MVP de *Machine Learning* para **classificação binária de eventos indesejáveis** em poços de petróleo offshore, a partir de dados reais de sensoriamento do **3W Dataset** da Petrobras. Todo o pipeline — do download dos dados à avaliação dos modelos — é executável de ponta a ponta (concebido para rodar no Google Colab sem upload manual, autenticação ou chaves de acesso).

📓 Notebook: [`indesejaveis.ipynb`](indesejaveis.ipynb)

---

## 1. Problema

Em poços offshore, sensores monitoram continuamente pressão, temperatura, vazão e estado de válvulas. Eventos indesejáveis (fechamento espúrio de válvula, formação de hidrato, golfada severa, etc.) causam perda de produção e riscos operacionais. O objetivo é, dado um trecho de leituras dos sensores, determinar se o poço está em condição **normal** ou em **anomalia**.

- **Tarefa:** classificação binária supervisionada.
- **Alvo:** derivado da coluna `class` do 3W — `0` = normal; qualquer valor ≠ 0 = anomalia.
- **Foco de avaliação:** F1, recall e AUC (não a acurácia), por causa do desbalanceamento das classes.

**Premissas assumidas:** classes desbalanceadas (anomalia é o evento raro); dados são séries temporais (exigem cuidado na divisão treino/teste para evitar vazamento); uso de um subconjunto do dataset para viabilizar execução leve.

## 2. Dados

**3W Dataset** — séries temporais multivariadas de sensores de poços offshore, com eventos rotulados por especialistas da Petrobras. Primeiro conjunto realista e público de eventos raros indesejáveis em poços de petróleo (lançado em 2019).

- Repositório oficial: https://github.com/petrobras/3W
- Espelho citável (Figshare): https://figshare.com/projects/3W_Dataset/251195
- UCI ML Repository: https://archive.ics.uci.edu/dataset/540/3w+dataset
- **Licença:** Creative Commons Attribution 4.0 (CC BY 4.0)

Cada instância é um arquivo Parquet; os subdiretórios de `dataset` são nomeados pelo código do tipo de evento (`0` = normal). O notebook usa a **API pública do GitHub** para baixar seletivamente um número reduzido de instâncias, sem transferência manual.

- **Evento estudado:** código **2** — *fechamento espúrio da DHSV* (válvula de segurança de subsuperfície), que enseja perda de produção.
- **Amostra:** 15 instâncias reais de evento + 15 instâncias normais.
- Rótulo por observação na coluna `class`; 27 variáveis de sensores detectadas.

## 3. Metodologia

### Análise exploratória (EDA)
- Cada instância tem ~12.000+ observações (1 leitura/segundo) e 30 colunas.
- **Valores ausentes expressivos:** 9 sensores com 100% de dados faltantes no subconjunto (falta de instrumentação em determinados poços); outros com ausência parcial.
- **Desbalanceamento:** ~84% normal / ~16% anomalia por observação.

### Preparação dos dados
- **Janelamento:** séries segmentadas em janelas fixas de `WINDOW = 300` observações (~5 min a 1 Hz).
- **Extração de atributos:** para cada sensor e janela, 4 estatísticas — média, desvio-padrão, mínimo e máximo.
- **Rótulo da janela:** anomalia quando ≥ 50% das observações são anômalas.
- Remoção das colunas 100% vazias → **matriz final de 1.533 janelas × 72 atributos** (18 sensores × 4 estatísticas).
- Imputação (mediana) e padronização feitas **dentro do pipeline** (previne vazamento).

### Divisão treino/teste (decisão metodológica central)
- **Divisão por instância, não por registro:** todas as janelas de um mesmo poço vão inteiramente para treino *ou* teste — nunca ambos. Evita o vazamento entre janelas vizinhas correlacionadas, que inflaria artificialmente as métricas.
- Estratificação pelo rótulo dominante de cada instância.
- Resultado: **1.089 janelas de treino / 444 de teste**, com proporção de anomalia praticamente igual (0,159 vs 0,162).

### Modelagem
Pipelines idênticos (imputação → padronização → estimador), com `class_weight="balanced"`:
1. **Baseline** — classe majoritária (`DummyClassifier`).
2. **Regressão Logística** — linear, interpretável.
3. **Random Forest** — captura interações não-lineares.

### Otimização
- `GridSearchCV` no Random Forest (critério F1, CV estratificada 4-fold) sobre `n_estimators`, `max_depth`, `min_samples_leaf`.

## 4. Resultados

| Modelo | F1 (anomalia) | ROC-AUC | PR-AUC | F1 (treino) |
|---|---|---|---|---|
| Baseline (classe majoritária) | 0,000 | 0,500 | 0,162 | 0,000 |
| Regressão Logística | **0,762** | 0,977 | 0,826 | 0,986 |
| Random Forest | 0,742 | 0,989 | **0,965** | 0,997 |
| Random Forest (tunado) | 0,738 | 0,987 | 0,950 | 0,991 |

Melhores hiperparâmetros do RF: `max_depth=8`, `min_samples_leaf=3`, `n_estimators=400` (F1 na CV: 0,989).

**Leitura dos resultados:**
- Ambos os modelos superam amplamente o baseline (F1 nulo), confirmando que a acurácia isolada seria enganosa.
- **F1 no limiar 0,5:** Regressão Logística (0,762) ≈ Random Forest (0,742).
- **Capacidade de separação:** Random Forest é claramente superior (PR-AUC 0,965 vs 0,826) — seu F1 menor no limiar padrão é questão de **calibração de limiar**, não de qualidade do modelo.
- A Regressão Logística obtém **recall 1,00** (detecta todos os eventos) ao custo de **precisão 0,62** (~40% de falsos alarmes entre os sinalizados).

**Sobre-ajuste:** gap treino−teste de ~0,22–0,25 em F1. Parte é overfitting real, mas parte reflete o **custo de medir a generalização corretamente** (poços não vistos). A discrepância entre F1 da CV (0,989) e do teste (~0,74) mostra que a CV estratificada por janela é otimista — o que valida a escolha da divisão por instância e aponta o `GroupKFold` como melhoria natural.

**Interpretabilidade:** poucos atributos concentram a importância no Random Forest. Coerente com a física do evento — no fechamento espúrio da DHSV, a pressão sobe a montante (P-PDG) e cai a jusante (P-TPT, P-MON-CKP).

## 5. Conclusão

O objetivo foi cumprido: classificar trechos de sinais quanto à condição normal ou anômala, com narrativa coerente do problema aos resultados e sem superestimação de desempenho.

**Modelo recomendado: Random Forest** — pela capacidade superior de separação das classes (PR-AUC 0,965) e pelo potencial de ajuste via calibração do limiar conforme o custo operacional (equilíbrio entre detectar eventos e conter falsos alarmes).

**Limitações assumidas:** subconjunto reduzido (15 instâncias/classe), único tipo de evento, CV não-agrupada na otimização, e descarte de ~1/3 dos sensores por ausência integral de dados.

**Próximos passos:** adotar `GroupKFold` na otimização; calibrar o limiar de decisão pelo custo operacional; ampliar número de instâncias e tipos de evento; enriquecer atributos (tendência, conteúdo espectral); avaliar *gradient boosting*.

## 6. Como executar

O notebook foi concebido para rodar no **Google Colab** sem configuração manual. Localmente, requer Python 3.10+ e:

```bash
pip install numpy pandas matplotlib scikit-learn pyarrow
```

Abra e execute `indesejaveis.ipynb` de cima para baixo. É necessário **acesso à internet** (o notebook baixa os dados via API pública do GitHub) — os arquivos ficam em cache local para reexecuções.

---

*Dados: 3W Dataset © Petrobras, licença CC BY 4.0.*
