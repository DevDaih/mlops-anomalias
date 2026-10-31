"""Testes da geração de registros do simulador (sem rede)."""

import importlib.util
from pathlib import Path

import numpy as np

from anomalias.data import FEATURES, MEANS, STDS

_spec = importlib.util.spec_from_file_location(
    "simular_trafego", Path(__file__).resolve().parents[1] / "scripts" / "simular_trafego.py"
)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

REF = {
    f: {"mean": float(m), "std": float(s)} for f, m, s in zip(FEATURES, MEANS, STDS, strict=True)
}


def test_drift_ramps_temperature_only():
    n, rps = 2000, 20
    base = sim.build_records("normal", REF, n, rps, seed=7)
    drifted = sim.build_records("drift", REF, n, rps, seed=7, ramp=50, shift=1.5)
    diff = drifted - base
    j = FEATURES.index("temperature")
    assert diff[0, j] == 0
    np.testing.assert_allclose(diff[-1, j], 1.5 * REF["temperature"]["std"])
    other = [i for i in range(len(FEATURES)) if i != j]
    assert np.all(diff[:, other] == 0)


def test_drift_extra_feature():
    drifted = sim.build_records(
        "drift", REF, 1000, 20, seed=7, ramp=10, drift_features=("current",)
    )
    base = sim.build_records("normal", REF, 1000, 20, seed=7)
    j = FEATURES.index("current")
    assert (drifted - base)[-1, j] > 0


def test_burst_changes_only_middle_third():
    n, rps = 900, 20
    base = sim.build_records("normal", REF, n, rps, seed=7)
    burst = sim.build_records("burst", REF, n, rps, seed=7)
    changed = np.any(burst != base, axis=1)
    assert not changed[: n // 3 - 1].any()
    assert not changed[2 * n // 3 + 1 :].any()
    assert changed[n // 3 : 2 * n // 3].mean() > 0.9


def test_burst_custom_start_and_duration():
    n, rps = 600, 20  # 30 s
    base = sim.build_records("normal", REF, n, rps, seed=7)
    burst = sim.build_records("burst", REF, n, rps, seed=7, burst_start=10, burst_duration=20)
    changed = np.any(burst != base, axis=1)
    assert not changed[: 10 * rps - 1].any()
    assert changed[10 * rps :].mean() > 0.9


def test_burst_args_validation(capsys):
    args = sim.parse_args(["--mode", "burst", "--burst-start", "100", "--burst-duration", "200"])
    assert (args.burst_start, args.burst_duration) == (100, 200)
    try:
        sim.parse_args(["--burst-duration", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("esperava SystemExit")


def test_records_centered_on_reference():
    ref = {f: {"mean": 100.0 + i, "std": 2.0} for i, f in enumerate(FEATURES)}
    X = sim.build_records("normal", ref, 4000, 20, seed=7)
    np.testing.assert_allclose(np.median(X, axis=0), [100, 101, 102, 103], atol=0.3)
