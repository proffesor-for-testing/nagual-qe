#!/usr/bin/env python3
"""
Train the FastGRNN router (query complexity -> vendor tier) on labelled queries.

Pipeline (run from the repository root):

    cargo run -q --example router_features --no-default-features --features kos -- \\
        models/router_queries.jsonl > /tmp/router_features.jsonl
    python3 models/train_fastgrnn.py --features /tmp/router_features.jsonl \\
        --output models/fastgrnn_router.json

`router_features` computes the 5 input features with the production `ComplexityEstimator`,
so the weights are trained on exactly what `VendorRouter` sees at runtime:

    [query_length, reasoning_demand, domain_specificity, structure, historical_accuracy]

Labels come from `models/router_queries.jsonl` (level + train/test split, rubric in
models/README.md). Targets are the level midpoints; the router's thresholds are 0.3 / 0.5 / 0.7.

The model is the exact forward pass of `src/router/fastgrnn.rs` for a single step from a zero
hidden state:

    z = sigmoid(W_z x + b_z)
    h = (zeta * (1 - z) + nu) * tanh(W_h x + b_h)
    y = sigmoid(W_o h + b_o)

(U_z / U_h multiply the zero initial state, so they are exported as zeros.)

This replaces an earlier script that trained on random synthetic features with incomplete
gradients (no gate updates, no activation derivatives); the resulting weights scored almost
every query 0.47-0.53. This version is dependency-free (pure Python); `--export-onnx` needs
torch + numpy.
"""

import argparse
import hashlib
import json
import math
import random

LEVEL_TARGET = {"low": 0.15, "medium": 0.40, "high": 0.60, "very_high": 0.85}
LEVELS = ["low", "medium", "high", "very_high"]


def level_of(score):
    """Same thresholds as ComplexityLevel::from_score."""
    if score < 0.3:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 0.7:
        return "high"
    return "very_high"


def is_cloud(level):
    """VendorSelector (default config): >= 0.5 routes to a cloud vendor."""
    return level in ("high", "very_high")


def sigmoid(v):
    if v < -60:
        return 0.0
    if v > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-v))


class Model:
    def __init__(self, input_dim, hidden_dim, zeta, nu, rng):
        self.n, self.h, self.zeta, self.nu = input_dim, hidden_dim, zeta, nu
        xi = math.sqrt(6.0 / (input_dim + hidden_dim))
        xo = math.sqrt(6.0 / (hidden_dim + 1))
        self.p = {
            "w_z": [[rng.uniform(-xi, xi) for _ in range(input_dim)] for _ in range(hidden_dim)],
            "b_z": [0.0] * hidden_dim,
            "w_h": [[rng.uniform(-xi, xi) for _ in range(input_dim)] for _ in range(hidden_dim)],
            "b_h": [0.0] * hidden_dim,
            "w_o": [rng.uniform(-xo, xo) for _ in range(hidden_dim)],
            "b_o": 0.0,
        }

    def forward(self, x):
        p = self.p
        z = [sigmoid(sum(w * xi for w, xi in zip(p["w_z"][j], x)) + p["b_z"][j]) for j in range(self.h)]
        t = [math.tanh(sum(w * xi for w, xi in zip(p["w_h"][j], x)) + p["b_h"][j]) for j in range(self.h)]
        s = [self.zeta * (1.0 - zj) + self.nu for zj in z]
        h = [sj * tj for sj, tj in zip(s, t)]
        y = sigmoid(sum(w * hj for w, hj in zip(p["w_o"], h)) + p["b_o"])
        return y, (z, t, s, h)

    def grads(self, batch, l2):
        p, H, N = self.p, self.h, self.n
        g = {
            "w_z": [[0.0] * N for _ in range(H)], "b_z": [0.0] * H,
            "w_h": [[0.0] * N for _ in range(H)], "b_h": [0.0] * H,
            "w_o": [0.0] * H, "b_o": 0.0,
        }
        loss = 0.0
        for x, target in batch:
            y, (z, t, s, h) = self.forward(x)
            loss += (y - target) ** 2
            g_o = 2.0 * (y - target) * y * (1.0 - y)
            g["b_o"] += g_o
            for j in range(H):
                g["w_o"][j] += g_o * h[j]
                dh = g_o * p["w_o"][j]
                da_h = dh * s[j] * (1.0 - t[j] ** 2)
                da_z = dh * t[j] * (-self.zeta) * z[j] * (1.0 - z[j])
                g["b_h"][j] += da_h
                g["b_z"][j] += da_z
                for i in range(N):
                    g["w_h"][j][i] += da_h * x[i]
                    g["w_z"][j][i] += da_z * x[i]
        m = float(len(batch))
        for k in ("w_z", "w_h"):
            for j in range(H):
                for i in range(N):
                    g[k][j][i] = g[k][j][i] / m + l2 * p[k][j][i]
        for k in ("b_z", "b_h"):
            g[k] = [v / m for v in g[k]]
        g["w_o"] = [v / m + l2 * w for v, w in zip(g["w_o"], p["w_o"])]
        g["b_o"] /= m
        return loss / m, g


def flat_keys(model):
    """(param key, index path) for every scalar, so Adam can run over a flat view."""
    keys = []
    for k, v in model.p.items():
        if isinstance(v, float):
            keys.append((k, ()))
        elif isinstance(v[0], list):
            keys += [(k, (j, i)) for j in range(len(v)) for i in range(len(v[0]))]
        else:
            keys += [(k, (j,)) for j in range(len(v))]
    return keys


def get(d, k, idx):
    v = d[k]
    for i in idx:
        v = v[i]
    return v


def put(d, k, idx, val):
    if not idx:
        d[k] = val
    elif len(idx) == 1:
        d[k][idx[0]] = val
    else:
        d[k][idx[0]][idx[1]] = val


def train(data, hidden, epochs, lr, l2, seed, zeta, nu):
    rng = random.Random(seed)
    model = Model(len(data[0][0]), hidden, zeta, nu, rng)
    keys = flat_keys(model)
    m1 = {kk: 0.0 for kk in keys}
    m2 = {kk: 0.0 for kk in keys}
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = None
    for step in range(1, epochs + 1):
        loss, g = model.grads(data, l2)  # full batch: the set is small
        for kk in keys:
            gv = get(g, *kk)
            m1[kk] = b1 * m1[kk] + (1 - b1) * gv
            m2[kk] = b2 * m2[kk] + (1 - b2) * gv * gv
            mhat = m1[kk] / (1 - b1 ** step)
            vhat = m2[kk] / (1 - b2 ** step)
            put(model.p, *kk, get(model.p, *kk) - lr * mhat / (math.sqrt(vhat) + eps))
    return model, loss


def evaluate(model, rows):
    preds = [model.forward(r["features"])[0] for r in rows]
    targets = [LEVEL_TARGET[r["level"]] for r in rows]
    n = len(rows)
    exact = sum(level_of(p) == r["level"] for p, r in zip(preds, rows)) / n
    tier = sum(is_cloud(level_of(p)) == is_cloud(r["level"]) for p, r in zip(preds, rows)) / n
    worst = max(abs(LEVELS.index(level_of(p)) - LEVELS.index(r["level"])) for p, r in zip(preds, rows))
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n
    mae = sum(abs(p - t) for p, t in zip(preds, targets)) / n
    return {
        "n": n, "mse": round(mse, 5), "mae": round(mae, 5),
        "level_accuracy": round(exact, 4), "local_vs_cloud_accuracy": round(tier, 4),
        "max_level_error": worst,
    }


def export(model, cfg, metrics, data_meta, out):
    H, N = model.h, model.n
    weights = {
        "w_z": [v for row in model.p["w_z"] for v in row],
        "u_z": [0.0] * (H * H),
        "b_z": model.p["b_z"],
        "w_h": [v for row in model.p["w_h"] for v in row],
        "u_h": [0.0] * (H * H),
        "b_h": model.p["b_h"],
        "w_o": model.p["w_o"],
        "b_o": [model.p["b_o"]],
        "zeta": model.zeta,
        "nu": model.nu,
    }
    doc = {"config": cfg, "weights": weights, "metrics": metrics, "data": data_meta}
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def export_onnx(json_path, onnx_path):
    import numpy as np
    import torch
    import torch.nn as nn

    doc = json.load(open(json_path))
    w, cfg = doc["weights"], doc["config"]
    H, N = cfg["hidden_dim"], cfg["input_dim"]

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            t = lambda v, shape: torch.tensor(np.array(v, dtype=np.float32).reshape(shape))
            self.w_z, self.b_z = t(w["w_z"], (H, N)), t(w["b_z"], (H,))
            self.w_h, self.b_h = t(w["w_h"], (H, N)), t(w["b_h"], (H,))
            self.w_o, self.b_o = t(w["w_o"], (1, H)), t(w["b_o"], (1,))

        def forward(self, x):
            z = torch.sigmoid(x @ self.w_z.T + self.b_z)
            h = (w["zeta"] * (1 - z) + w["nu"]) * torch.tanh(x @ self.w_h.T + self.b_h)
            return torch.sigmoid(h @ self.w_o.T + self.b_o)

    torch.onnx.export(Router(), torch.zeros(1, N), onnx_path, input_names=["features"],
                      output_names=["complexity"], dynamic_axes={"features": {0: "batch"}}, opset_version=13)
    print(f"ONNX model written to {onnx_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="/tmp/router_features.jsonl",
                    help="output of `cargo run --example router_features`")
    ap.add_argument("--output", default="models/fastgrnn_router.json")
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--export-onnx", metavar="PATH", help="also write an ONNX model (needs torch)")
    a = ap.parse_args()

    raw = open(a.features, "rb").read()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    data = [(r["features"], LEVEL_TARGET[r["level"]]) for r in train_rows]
    zeta, nu = 1.0, -0.001

    best = None
    for k in range(a.restarts):
        model, loss = train(data, a.hidden_dim, a.epochs, a.lr, a.l2, a.seed + k, zeta, nu)
        print(f"restart {k}: train mse {loss:.5f}")
        if best is None or loss < best[1]:  # selected on TRAIN loss only
            best = (model, loss)
    model = best[0]

    metrics = {"train": evaluate(model, train_rows), "test": evaluate(model, test_rows)}
    print(json.dumps(metrics, indent=2))
    cfg = {"input_dim": len(data[0][0]), "hidden_dim": a.hidden_dim, "output_dim": 1, "zeta": zeta, "nu": nu,
           "optimizer": "adam", "learning_rate": a.lr, "epochs": a.epochs, "l2": a.l2,
           "restarts": a.restarts, "seed": a.seed}
    data_meta = {"source": "models/router_queries.jsonl", "features_sha256": hashlib.sha256(raw).hexdigest(),
                 "train_rows": len(train_rows), "test_rows": len(test_rows),
                 "targets": LEVEL_TARGET, "selection": "lowest train mse across restarts"}
    export(model, cfg, metrics, data_meta, a.output)
    print(f"weights written to {a.output}")
    if a.export_onnx:
        export_onnx(a.output, a.export_onnx)


if __name__ == "__main__":
    main()
