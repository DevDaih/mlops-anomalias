# Projeto: Detecção de anomalias como serviço (mini plataforma MLOps)

## Objetivo
Portfólio de MLOps. Modelo IsolationForest (scikit-learn) treinado com dados
SINTÉTICOS com anomalias conhecidas, servido por FastAPI, com monitoramento
(Prometheus, Grafana, Alertmanager), CI/CD, retreino agendado com portão de
qualidade e infraestrutura como código (Terraform, provider Docker).

## Restrições
- Sem OpenAI e sem qualquer provedor de IA externo (a API nunca chama serviço
  externo; a única exceção é publicar o modelo no Hugging Face, veja a seção
  Hugging Face). Tudo roda local e de graça.
- Máquina de 2 núcleos e pouca RAM: imagens slim, nada pesado.
- Treino precisa levar segundos.
- Dados são sempre sintéticos e isso deve estar dito no README.

## Estrutura
- src/anomalias/     código Python (data.py, train.py, model_store.py, drift.py)
- api/               FastAPI (main.py, metrics.py, schemas.py)
- models/            artefatos versionados (model_vN.joblib + model_vN.json)
- monitoring/        prometheus/, grafana/ (provisioning), alertmanager/
- scripts/           bash e simulador de tráfego
- tests/             pytest
- infra/terraform/   ideia futura, ainda não existe
- .github/workflows/ CI existe (ci.yml)

## Convenções
- Python 3.11, type hints, ruff para lint, pytest para testes.
- Nomes de métricas Prometheus com prefixo `anomaly_api_`.
- Comentários e README em português; nomes de código em inglês.
- Antes de editar muitos arquivos, apresente um plano curto.
- Ao terminar uma tarefa: rode ruff e pytest e mostre o resultado.
- Nunca commitar segredos.

## Hugging Face
Exceção à regra de "sem provedor externo": é permitido PUBLICAR o modelo
treinado no Hugging Face Hub (scripts/exportar_hf.py --push), usando HF_TOKEN
do ambiente. A API continua sem depender de nenhum serviço externo em
execução, e o token nunca vai para o repositório nem para a imagem Docker.
