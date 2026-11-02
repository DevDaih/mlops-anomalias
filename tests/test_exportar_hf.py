"""Testes da exportação para o Hugging Face Hub (sem rede, tudo em tmp_path)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("skops")
pytest.importorskip("huggingface_hub")

import huggingface_hub  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from anomalias import model_store  # noqa: E402
from anomalias.data import generate_dataset  # noqa: E402
from anomalias.train import train  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exportar_hf", Path(__file__).resolve().parents[1] / "scripts" / "exportar_hf.py"
)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

TOKEN = "hf_TOKEN_SECRETO_DE_TESTE_123"


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models")
    model, metadata = train(n_samples=1500, seed=42, n_estimators=50)
    model_store.save_model(model, metadata, models_dir=path)
    return path


def test_export_gera_arquivos_e_equivalencia(models_dir, tmp_path):
    out = tmp_path / "hf_export"
    summary = exp.export(models_dir, out)

    assert summary["version"] == 1
    assert summary["n_checked"] == 1000
    assert summary["untrusted_types"], "esperava ao menos um tipo não confiável de sklearn/numpy"
    assert all(t.split(".")[0] in ("sklearn", "numpy") for t in summary["untrusted_types"])
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "model.skops", "model_v1.json"]
    # O JSON copiado é o mesmo do modelo ativo
    assert (out / "model_v1.json").read_text() == (models_dir / "model_v1.json").read_text()


def test_equivalencia_exata_com_seed_diferente_do_treino(models_dir, tmp_path):
    out = tmp_path / "hf"
    exp.export(models_dir, out)
    original, metadata = model_store.load_current(models_dir)
    reloaded, _ = exp.load_trusting_only_sklearn_numpy(out / "model.skops")
    X, _ = generate_dataset(300, 0.05, seed=metadata["data"]["seed"] + 99)
    assert (original.score_samples(X) == reloaded.score_samples(X)).all()


def test_equivalencia_falha_se_modelos_diferem(models_dir):
    original, metadata = model_store.load_current(models_dir)
    outro = IsolationForest(n_estimators=50, random_state=1).fit(generate_dataset(500, 0.05, 1)[0])
    with pytest.raises(exp.ExportError, match="diferem"):
        exp.check_equivalence(original, outro, metadata)


def test_falha_se_tipo_fora_de_sklearn_numpy(models_dir, tmp_path, monkeypatch):
    import skops.io

    monkeypatch.setattr(skops.io, "get_untrusted_types", lambda **_: ["os.system"])
    with pytest.raises(exp.ExportError, match="os.system"):
        exp.export(models_dir, tmp_path / "hf")


def test_cartao_contem_metricas_do_json(models_dir, tmp_path):
    out = tmp_path / "hf"
    exp.export(models_dir, out)
    card = (out / "README.md").read_text(encoding="utf-8")
    metadata = json.loads((out / "model_v1.json").read_text())

    for key in ("precision", "recall", "f1", "roc_auc"):
        assert f"{metadata['metrics'][key]:.4f}".replace(".", ",") in card
    assert str(metadata["data"]["n_samples"]) in card
    n_anom = round(metadata["data"]["n_test"] * metadata["hyperparams"]["contamination"])
    flat = " ".join(card.split())  # o texto quebra linha no meio das frases
    assert f"cerca de {n_anom} anomalias" in flat
    assert f"versão v{metadata['version']} do modelo do projeto" in flat
    assert "poucas dezenas" not in flat
    assert "{{" not in card
    assert card.startswith("---\nlibrary_name: sklearn\nlicense: mit\n")
    assert "SINTÉTICOS" in card


def test_cartao_rejeita_placeholder_desconhecido(models_dir):
    _, metadata = model_store.load_current(models_dir)
    with pytest.raises(exp.ExportError, match="inexistente"):
        exp.render_model_card("valor {{inexistente}}", metadata)


def test_sem_push_nao_chama_hub_nem_imprime_token(models_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    def boom(*a, **k):
        raise AssertionError("não deveria chamar o Hub sem --push")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    rc = exp.main(["--models-dir", str(models_dir), "--output", str(tmp_path / "hf")])
    captured = capsys.readouterr()
    assert rc == 0
    assert TOKEN not in captured.out + captured.err
    assert "Sem --push" in captured.out


def test_push_usa_token_do_ambiente_sem_imprimir(models_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    calls: dict = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, **kw):
            calls["repo"] = repo_id

        def upload_folder(self, **kw):
            calls["upload"] = kw

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    rc = exp.main(
        [
            "--models-dir", str(models_dir),
            "--output", str(tmp_path / "hf"),
            "--push",
            "--repo-id", "usuario/modelo",
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert rc == 0
    assert calls["token"] == TOKEN
    assert calls["repo"] == "usuario/modelo"
    assert calls["upload"]["repo_id"] == "usuario/modelo"
    assert TOKEN not in captured.out + captured.err


def test_erro_do_hub_nao_vaza_token(models_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    class FailingApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, **kw):
            raise RuntimeError(f"401 Unauthorized (Authorization: Bearer {TOKEN})")

    monkeypatch.setattr(huggingface_hub, "HfApi", FailingApi)
    rc = exp.main(
        ["--models-dir", str(models_dir), "--output", str(tmp_path / "hf"),
         "--push", "--repo-id", "usuario/modelo"]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert rc == 1
    assert TOKEN not in captured.out + captured.err
    assert "***" in captured.err


@pytest.mark.parametrize("argv_extra", [[], ["--repo-id", "usuario/modelo"]])
def test_push_exige_repo_id_e_token(models_dir, tmp_path, monkeypatch, capsys, argv_extra):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = exp.main(
        ["--models-dir", str(models_dir), "--output", str(tmp_path / "hf"), "--push", *argv_extra]
    )
    assert rc == 1
    assert "erro:" in capsys.readouterr().err
