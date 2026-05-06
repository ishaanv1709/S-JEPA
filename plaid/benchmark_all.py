"""
Vortaz Labs — PLAID Full Benchmark: World Model vs ML vs LLMs
================================================================
Comprehensive comparison on PLAID AirfRANS data:

  1. SYMBOLIC-JEPA (our world model)
  2. ML BASELINES (trained on same data):
     - Linear Regression
     - Random Forest
     - Gradient Boosting (XGBoost-like)
     - MLP (sklearn)
     - Deep MLP (PyTorch)
  3. LLM BASELINES (zero-shot via Groq API):
     - GPT-OSS 120B
     - GPT-OSS 20B
     - Llama 3.1 8B
     - Llama 4 Scout 17B

Every single prediction is logged to: plaid/results/<run_timestamp>/
Each model gets its own log file with per-sample outputs.

Usage: python plaid/benchmark_all.py
"""

import os
import sys
import time
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaid.dataset import PLAIDDataset, POOLED_DIM, ACTION_DIM

# ═══════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════

class BenchmarkLogger:
    """Logs every prediction to a dedicated results folder."""

    def __init__(self, run_dir: str, model_name: str):
        self.model_name = model_name
        self.model_dir = Path(run_dir) / model_name.replace(" ", "_").lower()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.model_dir / "predictions.jsonl"
        self.summary_path = self.model_dir / "summary.json"
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        self.predictions = []
        self.targets = []
        self.latencies = []
        self.sample_count = 0

        print(f"    Logging to: {self.model_dir}")

    def log_prediction(self, sample_idx, state, action, target,
                       prediction, latency_ms, extra=None):
        """Log a single prediction."""
        record = {
            "sample_idx": int(sample_idx),
            "state": state.tolist() if hasattr(state, 'tolist') else list(state),
            "action": action.tolist() if hasattr(action, 'tolist') else list(action),
            "target": target.tolist() if hasattr(target, 'tolist') else list(target),
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else list(prediction),
            "mae": float(np.mean(np.abs(np.array(prediction) - np.array(target)))),
            "latency_ms": float(latency_ms),
        }
        if extra:
            record.update(extra)

        self.log_file.write(json.dumps(record) + "\n")
        self.predictions.append(np.array(prediction))
        self.targets.append(np.array(target))
        self.latencies.append(latency_ms)
        self.sample_count += 1

    def finalize(self):
        """Compute and save summary metrics."""
        self.log_file.close()

        if not self.predictions:
            summary = {"model": self.model_name, "error": "no predictions"}
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            return summary

        preds = np.array(self.predictions)
        tgts = np.array(self.targets)

        mae = float(np.mean(np.abs(preds - tgts)))
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        mse = float(np.mean((preds - tgts) ** 2))

        # Per-feature MAE
        per_feature_mae = np.mean(np.abs(preds - tgts), axis=0).tolist()

        # R² score
        ss_res = np.sum((tgts - preds) ** 2)
        ss_tot = np.sum((tgts - tgts.mean(axis=0)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-8))

        # Cosine similarity (average across samples)
        cos_sims = []
        for p, t in zip(preds, tgts):
            norm_p = np.linalg.norm(p)
            norm_t = np.linalg.norm(t)
            if norm_p > 1e-8 and norm_t > 1e-8:
                cos_sims.append(float(np.dot(p, t) / (norm_p * norm_t)))
        avg_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

        summary = {
            "model": self.model_name,
            "n_samples": self.sample_count,
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "r2_score": r2,
            "cosine_similarity": avg_cos,
            "per_feature_mae": per_feature_mae,
            "avg_latency_ms": float(np.mean(self.latencies)),
            "total_latency_s": float(np.sum(self.latencies) / 1000),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also save raw predictions for later analysis
        np.savez_compressed(
            str(self.model_dir / "raw_predictions.npz"),
            predictions=preds,
            targets=tgts,
            latencies=np.array(self.latencies),
        )

        return summary


# ═══════════════════════════════════════════════════════════
# MODEL 1: SYMBOLIC-JEPA (OUR WORLD MODEL)
# ═══════════════════════════════════════════════════════════

def evaluate_symbolic_jepa(test_data, run_dir, device):
    """Evaluate the Symbolic-JEPA world model."""
    print(f"\n  {'='*55}")
    print(f"  MODEL: Symbolic-JEPA (Ours)")
    print(f"  {'='*55}")

    from plaid.symbolic_jepa import SymbolicJEPA, SymbolicDecoder

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_path = ckpt_dir / "plaid_symbolic_jepa.pth"

    if not ckpt_path.exists():
        print(f"    SKIP: No checkpoint at {ckpt_path}")
        print(f"    Run: python plaid/train_plaid.py first")
        return None

    model = SymbolicJEPA(input_dim=POOLED_DIM, action_dim=ACTION_DIM,
                         latent_dim=256, hidden_dim=256).to(device)
    decoder = SymbolicDecoder(latent_dim=256, output_dim=POOLED_DIM).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    model.eval()
    decoder.eval()
    print(f"    Loaded from {ckpt_path.name}")

    logger = BenchmarkLogger(run_dir, "Symbolic_JEPA")
    sample_idx = 0

    with torch.no_grad():
        for states, actions, targets, scores in tqdm(test_data, desc="    JEPA predict",
                                                      unit="batch",
                                                      bar_format='{l_bar}{bar:20}{r_bar}'):
            states = states.to(device)
            actions = actions.to(device)

            t0 = time.time()
            s_enc = model.encoder(states)
            s_pred = model.predictor(s_enc, actions)
            preds = decoder(s_pred)
            latency = (time.time() - t0) * 1000 / len(states)

            preds_np = preds.cpu().numpy()
            targets_np = targets.numpy()

            for i in range(len(states)):
                logger.log_prediction(
                    sample_idx, states[i].cpu().numpy(), actions[i].cpu().numpy(),
                    targets_np[i], preds_np[i], latency,
                )
                sample_idx += 1

                if sample_idx % 100 == 0:
                    print(f"    [{sample_idx}] predicted | latency: {latency:.2f}ms")

    summary = logger.finalize()
    print(f"    MAE: {summary['mae']:.6f} | R²: {summary['r2_score']:.4f} | "
          f"CosSim: {summary['cosine_similarity']:.4f}")
    return summary


# ═══════════════════════════════════════════════════════════
# MODEL 2-6: ML BASELINES
# ═══════════════════════════════════════════════════════════

def evaluate_ml_baselines(train_data, test_data, run_dir):
    """Train and evaluate traditional ML models."""
    print(f"\n  {'='*55}")
    print(f"  ML BASELINES (trained on same PLAID data)")
    print(f"  {'='*55}")

    # Collect train/test arrays
    print(f"    Collecting train/test arrays...")
    X_train, A_train, Y_train = [], [], []
    for states, actions, targets, scores in train_data:
        X_train.append(np.concatenate([states.numpy(), actions.numpy()], axis=1))
        Y_train.append(targets.numpy())
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    X_test, A_test_raw, Y_test, states_test = [], [], [], []
    for states, actions, targets, scores in test_data:
        X_test.append(np.concatenate([states.numpy(), actions.numpy()], axis=1))
        A_test_raw.append(actions.numpy())
        Y_test.append(targets.numpy())
        states_test.append(states.numpy())
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    A_test_np = np.concatenate(A_test_raw)
    states_test_np = np.concatenate(states_test)

    print(f"    Train: {X_train.shape} → {Y_train.shape}")
    print(f"    Test:  {X_test.shape} → {Y_test.shape}")

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    ml_models = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=12,
                                  n_jobs=-1, random_state=42)
        ),
        "Gradient_Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                      learning_rate=0.1, random_state=42)
        ),
        "MLP_Sklearn": MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            activation='relu', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            verbose=False,
        ),
    }

    summaries = {}

    for model_name, model in tqdm(ml_models.items(), desc="    ML models",
                                   unit="model",
                                   bar_format='{l_bar}{bar:20}{r_bar}'):
        print(f"\n    ── {model_name} ──")

        # Train
        t0 = time.time()
        print(f"    Training...")
        model.fit(X_train, Y_train)
        train_time = time.time() - t0
        print(f"    Trained in {train_time:.1f}s")

        # Predict
        logger = BenchmarkLogger(run_dir, f"ML_{model_name}")

        t0 = time.time()
        Y_pred = model.predict(X_test)
        total_pred_time = (time.time() - t0) * 1000
        per_sample_ms = total_pred_time / len(X_test)

        for i in range(len(X_test)):
            logger.log_prediction(
                i, states_test_np[i], A_test_np[i],
                Y_test[i], Y_pred[i], per_sample_ms,
                extra={"train_time_s": train_time}
            )
            if (i + 1) % 200 == 0:
                print(f"      [{i+1}/{len(X_test)}] logged")

        summary = logger.finalize()
        summary["train_time_s"] = train_time
        summaries[model_name] = summary
        print(f"    MAE: {summary['mae']:.6f} | R²: {summary['r2_score']:.4f} | "
              f"CosSim: {summary['cosine_similarity']:.4f} | "
              f"{per_sample_ms:.3f}ms/sample")

    # PyTorch Deep MLP
    print(f"\n    ── Deep_MLP_PyTorch ──")
    deep_mlp_summary = evaluate_deep_mlp(X_train, Y_train, X_test, Y_test,
                                          states_test_np, A_test_np, run_dir)
    if deep_mlp_summary:
        summaries["Deep_MLP_PyTorch"] = deep_mlp_summary

    return summaries


def evaluate_deep_mlp(X_train, Y_train, X_test, Y_test,
                       states_test, actions_test, run_dir):
    """Train and evaluate a deep MLP in PyTorch for fair GPU comparison."""
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    model = DeepMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {params:,}")

    # Train
    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    train_ds = torch.utils.data.TensorDataset(X_t, Y_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    t0 = time.time()
    model.train()
    for epoch in tqdm(range(30), desc="      DeepMLP train", unit="ep",
                      bar_format='{l_bar}{bar:20}{r_bar}'):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # Predict
    model.eval()
    logger = BenchmarkLogger(run_dir, "ML_Deep_MLP_PyTorch")

    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        t0 = time.time()
        Y_pred = model(X_te).cpu().numpy()
        total_ms = (time.time() - t0) * 1000
        per_sample = total_ms / len(X_test)

    for i in range(len(X_test)):
        logger.log_prediction(
            i, states_test[i], actions_test[i],
            Y_test[i], Y_pred[i], per_sample,
            extra={"train_time_s": train_time, "params": params}
        )

    summary = logger.finalize()
    summary["train_time_s"] = train_time
    summary["params"] = params
    print(f"    MAE: {summary['mae']:.6f} | R²: {summary['r2_score']:.4f} | "
          f"CosSim: {summary['cosine_similarity']:.4f}")
    return summary


# ═══════════════════════════════════════════════════════════
# MODEL 7-10: LLM BASELINES (GROQ API)
# ═══════════════════════════════════════════════════════════

def state_to_cfd_description(state_vec, action_vec, dataset):
    """Convert aggregated state + action into text for LLM."""
    # Denormalize if possible
    if hasattr(dataset, 'feat_mean'):
        raw = state_vec * dataset.feat_std + dataset.feat_mean
    else:
        raw = state_vec

    # Feature names: mean(u,v,dist,nx,ny,vx,vy,p,nut) + max(...) + std(...)
    feature_names = ["u_inlet", "v_inlet", "dist_to_airfoil",
                     "normal_x", "normal_y",
                     "velocity_x", "velocity_y", "pressure", "turb_viscosity"]

    desc_lines = ["Flow field summary (aggregated from CFD point cloud):"]

    # Mean features
    desc_lines.append("  Mean values:")
    for i, name in enumerate(feature_names):
        if i < len(raw):
            desc_lines.append(f"    {name}: {raw[i]:.4f}")

    # Max features
    desc_lines.append("  Max values:")
    for i, name in enumerate(feature_names):
        idx = 9 + i
        if idx < len(raw):
            desc_lines.append(f"    {name}: {raw[idx]:.4f}")

    # Action
    delta_aoa = action_vec[0]
    delta_re = action_vec[1]
    desc_lines.append(f"\nAction (configuration change):")
    desc_lines.append(f"  Delta Angle of Attack: {delta_aoa:.4f} (normalized)")
    desc_lines.append(f"  Delta Reynolds Number: {delta_re:.4f} (normalized)")

    return "\n".join(desc_lines)


def evaluate_llm_baselines(test_samples, dataset, run_dir, max_samples=100):
    """Evaluate LLMs via Groq API on PLAID predictions."""
    print(f"\n  {'='*55}")
    print(f"  LLM BASELINES (zero-shot via Groq API)")
    print(f"  Max samples per model: {max_samples}")
    print(f"  {'='*55}")

    # Check API keys
    from llm_baseline.llm_agent import _API_KEYS
    if not _API_KEYS:
        print(f"    SKIP: No GROQ_API_KEY found in .env")
        return {}

    print(f"    API keys available: {len(_API_KEYS)}")

    from llm_baseline.models_config import GROQ_MODELS
    from groq import Groq

    # Prepare test samples as text
    test_items = []
    for i, (state, action, target, score) in enumerate(test_samples):
        if i >= max_samples:
            break
        test_items.append({
            "idx": i,
            "state": state.numpy() if hasattr(state, 'numpy') else np.array(state),
            "action": action.numpy() if hasattr(action, 'numpy') else np.array(action),
            "target": target.numpy() if hasattr(target, 'numpy') else np.array(target),
            "description": state_to_cfd_description(
                state.numpy() if hasattr(state, 'numpy') else np.array(state),
                action.numpy() if hasattr(action, 'numpy') else np.array(action),
                dataset,
            ),
        })

    print(f"    Prepared {len(test_items)} test samples")

    summaries = {}
    key_idx = [0]

    def get_client():
        key = _API_KEYS[key_idx[0] % len(_API_KEYS)]
        key_idx[0] += 1
        return Groq(api_key=key)

    for model_key, model_config in tqdm(GROQ_MODELS.items(), desc="    LLM models",
                                         unit="model",
                                         bar_format='{l_bar}{bar:20}{r_bar}'):
        model_id = model_config["model_id"]
        model_name = model_config["display_name"]
        is_reasoning = model_config.get("is_reasoning", False)

        print(f"\n    ── {model_name} ({model_key}) ──")
        logger = BenchmarkLogger(run_dir, f"LLM_{model_key}")

        successes = 0
        failures = 0

        for item in tqdm(test_items, desc=f"      {model_name[:20]}", unit="sample",
                          leave=False,
                          bar_format='{l_bar}{bar:15}{r_bar}'):
            prompt = (
                f"You are a computational fluid dynamics expert. Given the following "
                f"flow field state and a configuration change, predict the new flow field state.\n\n"
                f"{item['description']}\n\n"
                f"Predict the NEW flow field after this configuration change.\n"
                f"Respond with ONLY a JSON object containing exactly {POOLED_DIM} float values as a list:\n"
                f'{{"predicted_state": [float, float, ..., float]}}\n'
                f"The list must have exactly {POOLED_DIM} values representing:\n"
                f"  [mean_u, mean_v, mean_dist, mean_nx, mean_ny, mean_vx, mean_vy, mean_p, mean_nut,\n"
                f"   max_u, max_v, max_dist, max_nx, max_ny, max_vx, max_vy, max_p, max_nut,\n"
                f"   std_u, std_v, std_dist, std_nx, std_ny, std_vx, std_vy, std_p, std_nut]"
            )

            messages = [
                {"role": "system", "content": "You are a physics simulation predictor. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ]

            t0 = time.time()
            try:
                client = get_client()

                if is_reasoning:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=1,
                        max_completion_tokens=2048,
                        top_p=1,
                        reasoning_effort="low",
                        stream=True,
                        stop=None,
                    )
                    parts = []
                    for chunk in completion:
                        parts.append(chunk.choices[0].delta.content or "")
                    content = "".join(parts)
                else:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=0.1,
                        max_completion_tokens=1024,
                        top_p=1,
                    )
                    content = completion.choices[0].message.content or ""

                latency_ms = (time.time() - t0) * 1000

                # Parse response
                prediction = parse_llm_cfd_response(content, POOLED_DIM)

                if prediction is not None:
                    logger.log_prediction(
                        item["idx"], item["state"], item["action"],
                        item["target"], prediction, latency_ms,
                        extra={"raw_response": content[:500], "parse_success": True}
                    )
                    successes += 1
                else:
                    # Failed parse — log zeros as prediction
                    logger.log_prediction(
                        item["idx"], item["state"], item["action"],
                        item["target"], np.zeros(POOLED_DIM), latency_ms,
                        extra={"raw_response": content[:500], "parse_success": False}
                    )
                    failures += 1

            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                logger.log_prediction(
                    item["idx"], item["state"], item["action"],
                    item["target"], np.zeros(POOLED_DIM), latency_ms,
                    extra={"error": str(e), "parse_success": False}
                )
                failures += 1

                # Rate limit handling
                if "rate" in str(e).lower() or "429" in str(e):
                    print(f"      Rate limited — waiting 10s...")
                    time.sleep(10)

            total = successes + failures
            if total % 10 == 0 and total > 0:
                print(f"      [{total}/{len(test_items)}] "
                      f"success={successes} fail={failures} "
                      f"({100*successes/total:.0f}%)")

            # Small delay to avoid rate limits
            time.sleep(0.5)

        summary = logger.finalize()
        summary["successes"] = successes
        summary["failures"] = failures
        summary["parse_rate"] = successes / max(successes + failures, 1)
        summaries[model_key] = summary

        print(f"    MAE: {summary['mae']:.6f} | R²: {summary['r2_score']:.4f} | "
              f"CosSim: {summary['cosine_similarity']:.4f} | "
              f"Parse rate: {summary['parse_rate']:.0%}")

    return summaries


def parse_llm_cfd_response(content: str, expected_dim: int):
    """Extract predicted state vector from LLM response."""
    # Try JSON parse
    try:
        data = json.loads(content)
        if "predicted_state" in data:
            vals = data["predicted_state"]
            if len(vals) == expected_dim:
                return np.array(vals, dtype=np.float32)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from text
    json_match = re.search(r'\{[^{}]*"predicted_state"\s*:\s*\[([^\]]+)\][^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            nums = [float(x.strip()) for x in json_match.group(1).split(",")]
            if len(nums) == expected_dim:
                return np.array(nums, dtype=np.float32)
        except ValueError:
            pass

    # Try extracting any list of numbers
    num_pattern = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
    if len(num_pattern) >= expected_dim:
        try:
            nums = [float(x) for x in num_pattern[:expected_dim]]
            return np.array(nums, dtype=np.float32)
        except ValueError:
            pass

    return None


# ═══════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — PLAID Full Benchmark")
    print("  World Model vs ML Baselines vs LLMs")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent / "results" / f"benchmark_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results: {results_dir}\n")

    # Check/generate data
    data_dir = str(Path(__file__).resolve().parent / "data")
    if not Path(data_dir).exists() or not (Path(data_dir) / "airfrans_processed.npz").exists():
        print(f"  No PLAID data found. Generating synthetic AirfRANS data...\n")
        from plaid.download_data import generate_synthetic_airfrans
        generate_synthetic_airfrans(data_dir, n_sims=1000)

    # Load dataset
    print(f"\n  Loading PLAID dataset...")
    dataset = PLAIDDataset(data_dir, max_pairs=5000)

    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds):,} | Test: {len(test_ds):,}")

    # Save dataset info
    dataset_info = {
        "n_simulations": dataset.n_sims,
        "n_pairs_total": len(dataset),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "feature_dim": POOLED_DIM,
        "action_dim": ACTION_DIM,
        "timestamp": timestamp,
    }
    with open(results_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    all_summaries = {}

    # ── 1. Symbolic-JEPA ──
    jepa_summary = evaluate_symbolic_jepa(test_loader, str(results_dir), device)
    if jepa_summary:
        all_summaries["Symbolic_JEPA"] = jepa_summary

    # ── 2. ML Baselines ──
    ml_summaries = evaluate_ml_baselines(train_loader, test_loader, str(results_dir))
    all_summaries.update({f"ML_{k}": v for k, v in ml_summaries.items()})

    # ── 3. LLM Baselines ──
    # Collect test samples for LLM evaluation
    test_samples = []
    for states, actions, targets, scores in test_loader:
        for i in range(len(states)):
            test_samples.append((states[i], actions[i], targets[i], scores[i]))

    llm_summaries = evaluate_llm_baselines(
        test_samples, dataset, str(results_dir), max_samples=50
    )
    all_summaries.update({f"LLM_{k}": v for k, v in llm_summaries.items()})

    # ══════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════
    total_time = time.time() - t0

    print(f"\n\n  {'='*75}")
    print(f"  PLAID BENCHMARK — FINAL RESULTS")
    print(f"  {'='*75}")

    print(f"\n  {'Model':<30} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'CosSim':>8} {'ms/sample':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

    # Sort by MAE (best first)
    sorted_models = sorted(all_summaries.items(),
                           key=lambda x: x[1].get('mae', 999))

    for rank, (name, s) in enumerate(sorted_models, 1):
        marker = " ★" if name == "Symbolic_JEPA" else ""
        print(f"  {rank}. {name:<28} "
              f"{s.get('mae', 0):>10.6f} "
              f"{s.get('rmse', 0):>10.6f} "
              f"{s.get('r2_score', 0):>8.4f} "
              f"{s.get('cosine_similarity', 0):>8.4f} "
              f"{s.get('avg_latency_ms', 0):>10.3f}"
              f"{marker}")

    # Save final comparison
    comparison = {
        "timestamp": timestamp,
        "total_time_s": total_time,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "models": all_summaries,
        "ranking": [name for name, _ in sorted_models],
    }
    comparison_path = results_dir / "final_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Also save a human-readable report
    report_path = results_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("  VORTAZ LABS — PLAID Benchmark Report\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Total time: {total_time:.1f}s\n")
        f.write("=" * 75 + "\n\n")

        f.write(f"{'Rank':<5} {'Model':<30} {'MAE':>10} {'RMSE':>10} "
                f"{'R²':>8} {'CosSim':>8} {'Latency':>10}\n")
        f.write("-" * 81 + "\n")

        for rank, (name, s) in enumerate(sorted_models, 1):
            marker = " ★ OURS" if name == "Symbolic_JEPA" else ""
            f.write(f"{rank:<5} {name:<30} "
                    f"{s.get('mae', 0):>10.6f} "
                    f"{s.get('rmse', 0):>10.6f} "
                    f"{s.get('r2_score', 0):>8.4f} "
                    f"{s.get('cosine_similarity', 0):>8.4f} "
                    f"{s.get('avg_latency_ms', 0):>10.3f}ms"
                    f"{marker}\n")

        f.write(f"\nTotal models evaluated: {len(sorted_models)}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"LLM samples: 50 (rate-limited)\n")

    print(f"\n  Total benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  All outputs saved to: {results_dir}")
    print(f"  Files per model:")
    print(f"    predictions.jsonl  — every single prediction logged")
    print(f"    summary.json       — aggregated metrics")
    print(f"    raw_predictions.npz — numpy arrays for analysis")
    print(f"  Global:")
    print(f"    final_comparison.json — side-by-side ranking")
    print(f"    report.txt            — human-readable report")
    print(f"    dataset_info.json     — dataset configuration")


if __name__ == "__main__":
    main()
