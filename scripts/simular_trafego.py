"""Simulador de tráfego para a API de anomalias (dados SINTÉTICOS).

Lê as reference_stats de GET /model e gera registros em torno delas:

    normal  ~5% de anomalias injetadas (generate_dataset, seed diferente da do treino)
    drift   a média da temperature (e de --features) desliza em rampa até --shift
            desvios, ao longo de --ramp segundos
    burst   tráfego normal com rajada de anomalias (--anomaly-rate) no terço
            central da duração

Exemplo:
    python scripts/simular_trafego.py --mode drift --duration 90 --rps 20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomalias.data import FEATURES, MEANS, STDS, generate_dataset  # noqa: E402

BASE_CONTAMINATION = 0.05
MODES = ("normal", "drift", "burst")


def to_reference(X: np.ndarray, reference_stats: dict[str, dict[str, float]]) -> np.ndarray:
    """Reescala dados do gerador (MEANS/STDS) para a média/desvio da referência do modelo."""
    ref_mean = np.array([reference_stats[f]["mean"] for f in FEATURES])
    ref_std = np.array([reference_stats[f]["std"] for f in FEATURES])
    return ref_mean + (X - MEANS) / STDS * ref_std


def build_records(
    mode: str,
    reference_stats: dict[str, dict[str, float]],
    n: int,
    rps: float,
    seed: int,
    ramp: float = 60.0,
    shift: float = 1.5,
    drift_features: tuple[str, ...] = (),
    anomaly_rate: float = 0.4,
) -> np.ndarray:
    """Gera n registros (n x len(FEATURES)), um por requisição, em ordem de envio."""
    X, _ = generate_dataset(n, BASE_CONTAMINATION, seed)
    X = to_reference(X, reference_stats)
    t = np.arange(n) / rps  # instante de envio de cada registro, em segundos

    if mode == "drift":
        frac = np.clip(t / ramp, 0.0, 1.0) if ramp > 0 else np.ones(n)
        for name in ("temperature", *drift_features):
            j = FEATURES.index(name)
            X[:, j] += frac * shift * reference_stats[name]["std"]
    elif mode == "burst":
        in_burst = (t >= t[-1] / 3) & (t < 2 * t[-1] / 3)
        # Pool com a taxa alta de anomalias; entra no lugar dos registros da rajada
        burst_pool, _ = generate_dataset(n, anomaly_rate, seed + 1)
        X[in_burst] = to_reference(burst_pool, reference_stats)[in_burst]
    return X


async def _send(client: httpx.AsyncClient, record: dict[str, float], stats: dict[str, Any]) -> None:
    start = time.perf_counter()
    try:
        resp = await client.post("/predict", json=record)
        stats["latencies"].append(time.perf_counter() - start)
        if resp.status_code == 200:
            stats["sent"] += 1
            stats["flagged"] += int(resp.json()["predictions"][0]["is_anomaly"])
        else:
            stats["errors"] += 1
    except httpx.HTTPError:
        stats["errors"] += 1


async def run(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(base_url=args.url, timeout=5.0) as client:
        try:
            info = (await client.get("/model")).raise_for_status().json()
        except httpx.HTTPError as exc:
            print(f"Não consegui ler GET /model em {args.url}: {exc}", file=sys.stderr)
            return 1

        seed = args.seed
        if seed == info["data"]["seed"]:
            seed += 1000
            print(f"seed igual à do treino; usando {seed}")

        n = max(1, int(args.duration * args.rps))
        features = tuple(f for f in args.features.split(",") if f)
        records = build_records(
            args.mode,
            info["reference_stats"],
            n,
            args.rps,
            seed,
            args.ramp,
            args.shift,
            features,
            args.anomaly_rate,
        )
        print(f"modo={args.mode} duração={args.duration}s rps={args.rps} registros={n}")

        stats: dict[str, Any] = {"sent": 0, "errors": 0, "flagged": 0, "latencies": []}
        tasks: list[asyncio.Task[None]] = []
        start = time.perf_counter()
        for i, row in enumerate(records):
            delay = start + i / args.rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            record = dict(zip(FEATURES, map(float, row), strict=True))
            tasks.append(asyncio.create_task(_send(client, record, stats)))
        await asyncio.gather(*tasks)

    lat = np.array(stats["latencies"]) * 1000
    p50, p95 = np.percentile(lat, [50, 95]) if lat.size else (float("nan"),) * 2
    print(
        f"enviados={stats['sent']} erros={stats['errors']} "
        f"marcados_como_anomalia={stats['flagged']}"
    )
    print(f"latência no cliente: p50={p50:.1f} ms  p95={p95:.1f} ms")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simula tráfego sintético para a API.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--mode", choices=MODES, default="normal")
    p.add_argument("--duration", type=float, default=60.0, help="segundos")
    p.add_argument("--rps", type=float, default=20.0, help="requisições por segundo")
    p.add_argument("--seed", type=int, default=123, help="seed do tráfego (≠ seed do treino)")
    p.add_argument(
        "--ramp", type=float, default=60.0, help="drift: segundos até o deslocamento total"
    )
    p.add_argument("--shift", type=float, default=1.5, help="drift: desvios do deslocamento final")
    p.add_argument("--features", default="", help="drift: outras features, separadas por vírgula")
    p.add_argument("--anomaly-rate", type=float, default=0.4, help="burst: taxa na rajada")
    args = p.parse_args(argv)

    if args.duration <= 0 or args.rps <= 0:
        p.error("--duration e --rps devem ser positivos")
    if not 0 <= args.anomaly_rate < 0.5:
        p.error("--anomaly-rate deve estar em [0, 0.5)")
    unknown = [f for f in args.features.split(",") if f and f not in FEATURES]
    if unknown:
        p.error(f"features desconhecidas: {unknown}; válidas: {FEATURES}")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
