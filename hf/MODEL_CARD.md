---
library_name: sklearn
license: mit
tags:
  - skops
  - anomaly-detection
  - tabular-classification
  - synthetic-data
---

# Detector de anomalias em leituras de sensores (IsolationForest, dados sintéticos)

## O que é

Um `IsolationForest` do scikit-learn (versão v{{version}} do modelo do projeto) que marca como anomalia leituras
de sensores industriais fora do padrão. Faz parte de uma mini plataforma de MLOps de portfólio
(API FastAPI, Prometheus, Grafana, CI/CD). O foco do projeto é o que existe ao redor do modelo,
não o modelo em si.

**Este modelo foi treinado e avaliado apenas com dados sintéticos.**

## Dados

Todos os dados são **SINTÉTICOS**, gerados por código (`src/anomalias/data.py` no repositório).

- 4 features: `temperature` (°C), `pressure` (bar), `vibration` (mm/s) e `current` (A), nessa ordem.
- Operação normal: normal multivariada com correlação leve entre as features.
- 3 tipos de anomalia injetados de forma controlada: `spike` (uma feature salta 6 a 10 desvios),
  `out_of_range` (várias features fora da faixa ao mesmo tempo) e `improbable_combo`
  (temperatura alta com corrente baixa e vibração alta, contrariando a correlação normal).
- Seed {{seed}}, {{n_samples}} amostras no total, {{contamination_pct}}% de anomalias; {{n_train}} amostras
  de treino e {{n_test}} de teste (divisão estratificada).

## Métricas

Medidas no conjunto de teste sintético ({{n_test}} amostras), a partir do `model_v{{version}}.json`
que acompanha este modelo:

| Métrica   | Valor |
|-----------|-------|
| Precisão  | {{precision}} |
| Recall    | {{recall}} |
| F1        | {{f1}} |
| ROC-AUC   | {{roc_auc}} |

Esses números valem só para o cenário sintético descrito acima (veja as limitações).

## Como usar

O modelo está no formato `skops` (não use `pickle`/`joblib` para carregar arquivos de terceiros).
Revise os tipos não confiáveis antes de confiar neles:

```python
import numpy as np
from skops.io import get_untrusted_types, load

tipos = get_untrusted_types(file="model.skops")
print(tipos)  # só tipos de sklearn e numpy são esperados; revise antes de confiar
model = load("model.skops", trusted=tipos)

# colunas, nessa ordem: temperature, pressure, vibration, current
X = np.array([[70.0, 5.0, 2.0, 15.0]])

raw = model.score_samples(X)
score = -raw  # maior = mais anômalo
is_anomaly = (raw - model.offset_) < 0  # mesmo critério de model.predict
```

O `model_v{{version}}.json` traz hiperparâmetros (`n_estimators` = {{n_estimators}},
`contamination` = {{contamination}}), métricas e `reference_stats` (média e desvio das features
usadas na detecção de drift).

## Uso pretendido e fora de escopo

- **Pretendido:** estudo, demonstração e testes de uma plataforma de MLOps (servir, monitorar,
  retreinar), e como exemplo de exportação segura de um modelo scikit-learn com `skops`.
- **Fora de escopo:** qualquer decisão sobre equipamentos, processos ou pessoas reais; detecção de
  anomalias em dados reais de sensores; uso como referência de desempenho esperado em produção.

## Limitações

- **O ROC-AUC de ~0,9996 não indica desempenho real.** Reflete um problema sintético fácil, com
  anomalias injetadas bem afastadas dos dados normais.
- **O modelo nunca viu dados reais.** Treino e teste saem do mesmo gerador; não há garantia nenhuma
  sobre como ele se comporta em sensores de verdade, com ruído, deriva e falhas diferentes das três
  simuladas.
- **A contaminação de 5% é conhecida no treino.** O hiperparâmetro `contamination` foi definido com
  a proporção real de anomalias dos dados sintéticos. Em produção essa proporção não é conhecida, e
  um valor errado desloca o limiar de decisão (`offset_`).
- **Precisão, recall e F1 vêm de um conjunto de teste pequeno** ({{n_test}} amostras, cerca de
  {{n_anomalies_test}} anomalias), então têm incerteza que não foi medida.
- **O monitor de drift do projeto só detecta mudança de média** das features (em desvios padrão da
  referência); não detecta mudança de variância, de correlação nem de forma da distribuição.

## Repositório

Código, API, monitoramento e CI: <https://github.com/devdaih/mlops-anomalias>
