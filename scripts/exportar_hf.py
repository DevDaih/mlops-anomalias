"""Exporta o modelo ativo para o formato skops e, opcionalmente, publica no Hugging Face Hub.

Uso:
    python scripts/exportar_hf.py                      # só gera hf_export/ (padrão)
    python scripts/exportar_hf.py --push --repo-id USUARIO/NOME

Requer `pip install -r requirements-hf.txt`. Com --push, o token vem da variável de
ambiente HF_TOKEN e nunca é impresso. Os dados do projeto são sintéticos.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from anomalias.data import generate_dataset
from anomalias.model_store import DEFAULT_MODELS_DIR, load_current
from anomalias.train import score_and_label

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "hf" / "MODEL_CARD.md"
MODEL_FILE = "model.skops"
N_CHECK = 1000
# Só tipos destes pacotes podem ser confiados ao carregar o arquivo skops
TRUSTED_ROOTS = ("sklearn", "numpy")


class ExportError(Exception):
    """Falha esperada da exportação (mensagem já pronta para o usuário)."""


def _redact(text: str, token: str | None) -> str:
    return text.replace(token, "***") if token else text


def save_skops(model: Any, path: Path) -> None:
    from skops.io import dump

    dump(model, path)


def load_trusting_only_sklearn_numpy(path: Path) -> tuple[Any, list[str]]:
    """Recarrega o .skops confiando só em tipos de scikit-learn/numpy.

    Retorna (modelo, tipos não confiáveis que apareceram). Levanta ExportError se
    aparecer qualquer tipo fora de TRUSTED_ROOTS.
    """
    from skops.io import get_untrusted_types, load

    untrusted = sorted(get_untrusted_types(file=path))
    outside = [t for t in untrusted if t.split(".")[0] not in TRUSTED_ROOTS]
    if outside:
        raise ExportError(f"tipos não confiáveis fora de sklearn/numpy: {outside}")
    return load(path, trusted=untrusted), untrusted


def check_equivalence(
    original: Any, reloaded: Any, metadata: dict[str, Any], n: int = N_CHECK
) -> int:
    """Confere que scores e rótulos do modelo recarregado são IDÊNTICOS aos do original.

    Usa dados sintéticos com seed diferente da do treino. Retorna o número de registros.
    """
    seed = int(metadata["data"]["seed"]) + 1
    contamination = float(metadata["hyperparams"]["contamination"])
    X, _ = generate_dataset(n, contamination, seed)
    scores_a, labels_a = score_and_label(original, X)
    scores_b, labels_b = score_and_label(reloaded, X)
    if not np.array_equal(scores_a, scores_b):
        raise ExportError("scores do modelo recarregado diferem do original")
    if not np.array_equal(labels_a, labels_b):
        raise ExportError("rótulos do modelo recarregado diferem do original")
    return n


def _pt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def render_model_card(template: str, metadata: dict[str, Any]) -> str:
    """Preenche o modelo de cartão; todos os números vêm do JSON de metadados."""
    metrics, data, hyper = metadata["metrics"], metadata["data"], metadata["hyperparams"]
    values = {
        "version": str(metadata["version"]),
        "seed": str(data["seed"]),
        "n_samples": str(data["n_samples"]),
        "n_train": str(data["n_train"]),
        "n_test": str(data["n_test"]),
        "n_anomalies_test": str(round(data["n_test"] * float(hyper["contamination"]))),
        "n_estimators": str(hyper["n_estimators"]),
        "contamination": _pt(float(hyper["contamination"]), 2),
        "contamination_pct": f"{float(hyper['contamination']) * 100:g}".replace(".", ","),
        "precision": _pt(metrics["precision"]),
        "recall": _pt(metrics["recall"]),
        "f1": _pt(metrics["f1"]),
        "roc_auc": _pt(metrics["roc_auc"]),
    }

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ExportError(f"placeholder desconhecido no cartão: {{{{{key}}}}}")
        return values[key]

    return re.sub(r"\{\{(\w+)\}\}", sub, template)


def export(models_dir: Path, output: Path, template: Path = DEFAULT_TEMPLATE) -> dict[str, Any]:
    """Gera os arquivos em `output` e devolve um resumo do que foi feito."""
    model, metadata = load_current(models_dir)
    version = int(metadata["version"])
    output.mkdir(parents=True, exist_ok=True)

    model_path = output / MODEL_FILE
    save_skops(model, model_path)
    reloaded, untrusted = load_trusting_only_sklearn_numpy(model_path)
    n_checked = check_equivalence(model, reloaded, metadata)

    meta_name = f"model_v{version}.json"
    shutil.copyfile(Path(models_dir) / meta_name, output / meta_name)

    card = render_model_card(template.read_text(encoding="utf-8"), metadata)
    (output / "README.md").write_text(card, encoding="utf-8")

    return {
        "version": version,
        "untrusted_types": untrusted,
        "n_checked": n_checked,
        "files": sorted(p.name for p in output.iterdir()),
    }


def push(output: Path, repo_id: str, token: str) -> None:
    import huggingface_hub

    api = huggingface_hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(output), repo_id=repo_id, repo_type="model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta o modelo ativo para o formato skops.")
    parser.add_argument("--output", type=Path, default=Path("hf_export"))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--push", action="store_true", help="Publica no Hub (usa HF_TOKEN).")
    parser.add_argument("--repo-id", help="Repositório no Hub, no formato usuario/nome.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN") if args.push else None
    try:
        if args.push and not args.repo_id:
            raise ExportError("--push exige --repo-id usuario/nome")
        if args.push and not token:
            raise ExportError("--push exige a variável de ambiente HF_TOKEN")

        summary = export(args.models_dir, args.output, args.template)
        print(f"Modelo ativo: v{summary['version']}")
        print("Tipos não confiáveis (todos de sklearn/numpy, confiados no recarregamento):")
        for t in summary["untrusted_types"]:
            print(f"  - {t}")
        print(
            f"Equivalência: scores e rótulos IDÊNTICOS em {summary['n_checked']} registros "
            "sintéticos (seed diferente da do treino)"
        )
        print(f"Arquivos em {args.output}/: {', '.join(summary['files'])}")

        if args.push:
            push(args.output, args.repo_id, token)
            print(f"Publicado em https://huggingface.co/{args.repo_id}")
        else:
            print("Sem --push: nada foi enviado ao Hub.")
    except ExportError as exc:
        print(f"erro: {_redact(str(exc), token)}", file=sys.stderr)
        return 1
    except Exception as exc:  # falha de rede/autenticação do Hub etc.
        print(f"erro: {type(exc).__name__}: {_redact(str(exc), token)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
