"""
Vortaz Labs — PLAID AirfRANS PyTorch Dataset
===============================================
Loads AirfRANS simulations as transition pairs for Symbolic-JEPA.

Each sample:
  state:      Aggregated point cloud features (pooled to fixed dim)
  action:     [delta_AoA, delta_Reynolds] (change in flight conditions)
  next_state: Aggregated features of the next simulation

Aggregation: Mean + Max pooling of point-level features → fixed 256D vector
This makes the representation discretization-invariant (different meshes → same dim).

Usage:
  from plaid.dataset import PLAIDDataset
  ds = PLAIDDataset("plaid/data")
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


# Features
INPUT_DIM = 5   # u, v, dist, nx, ny
TARGET_DIM = 4  # vel_x, vel_y, pressure, nu_t
ACTION_DIM = 2  # delta_AoA, delta_reynolds (normalized)

# After pooling: mean(5+4) + max(5+4) + std(5+4) = 27
POOLED_DIM = 27  # 9 * 3 (mean, max, std of concatenated input+target)


class PLAIDDataset(Dataset):
    """
    PyTorch Dataset for PLAID AirfRANS — transition pairs.

    Creates pairs (sim_i, sim_j) where the "action" is the change
    in flight conditions (AoA, Reynolds) between them.

    This teaches the Symbolic-JEPA to predict how flow fields
    change when an airfoil configuration changes.
    """

    def __init__(self, data_dir: str, max_pairs: int = 5000,
                 normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.sim_dir = self.data_dir / "simulations"

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load simulation index
        npz_path = self.data_dir / "airfrans_processed.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            self.aoas = data["aoas"]
            self.reynolds = data["reynolds"]
            self.n_sims = int(data["n_sims"])
            self.types = data["types"]
        else:
            raise FileNotFoundError(f"No processed data at {npz_path}. Run download_data.py first.")

        print(f"  PLAID Dataset: {self.n_sims} simulations")

        # Pre-compute aggregated features for each simulation
        print(f"  Aggregating point clouds to fixed-dim vectors...")
        self.features = []
        for i in tqdm(range(self.n_sims), desc="  Aggregating", unit="sim",
                      bar_format='{l_bar}{bar:30}{r_bar}'):
            sim_path = self.sim_dir / f"sim_{i:04d}.npz"
            sim = np.load(sim_path)
            feat = self._aggregate(sim["inputs"], sim["targets"])
            self.features.append(feat)

        self.features = np.array(self.features, dtype=np.float32)  # (n_sims, POOLED_DIM)

        # Create transition pairs
        print(f"  Creating transition pairs...")
        self.pairs = self._create_pairs(max_pairs)
        print(f"  Pairs: {len(self.pairs)}")

        # Normalize
        if normalize:
            self.feat_mean = self.features.mean(axis=0)
            self.feat_std = self.features.std(axis=0)
            self.feat_std[self.feat_std < 1e-6] = 1.0
            self.features = (self.features - self.feat_mean) / self.feat_std

            self.aoa_mean = self.aoas.mean()
            self.aoa_std = max(self.aoas.std(), 1e-6)
            self.re_mean = self.reynolds.mean()
            self.re_std = max(self.reynolds.std(), 1e-6)

    def _aggregate(self, inputs, targets):
        """
        Aggregate variable-length point cloud to fixed-dim vector.
        Uses mean + max + std pooling for discretization invariance.
        """
        combined = np.concatenate([inputs, targets], axis=1)  # (N, 9)

        mean_pool = combined.mean(axis=0)   # (9,)
        max_pool = combined.max(axis=0)     # (9,)
        std_pool = combined.std(axis=0)     # (9,)

        return np.concatenate([mean_pool, max_pool, std_pool])  # (27,)

    def _create_pairs(self, max_pairs):
        """Create transition pairs between simulations."""
        rng = np.random.RandomState(42)
        pairs = []

        for _ in range(max_pairs):
            i = rng.randint(0, self.n_sims)
            j = rng.randint(0, self.n_sims)
            if i != j:
                pairs.append((i, j))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]

        state = torch.tensor(self.features[i], dtype=torch.float32)
        next_state = torch.tensor(self.features[j], dtype=torch.float32)

        # Action = change in flight conditions (normalized)
        delta_aoa = (self.aoas[j] - self.aoas[i]) / max(self.aoa_std, 1e-6) if hasattr(self, 'aoa_std') else (self.aoas[j] - self.aoas[i]) / 10.0
        delta_re = (self.reynolds[j] - self.reynolds[i]) / max(self.re_std, 1e-6) if hasattr(self, 're_std') else (self.reynolds[j] - self.reynolds[i]) / 1e6

        action = torch.tensor([delta_aoa, delta_re], dtype=torch.float32)

        # Score: negative of MSE between predictions (for critic training)
        score = -float(np.mean((self.features[j] - self.features[i])**2))

        return state, action, next_state, torch.tensor(score, dtype=torch.float32)

    def get_naca4_indices(self):
        """Get indices of NACA 4-digit simulations (for training)."""
        return [i for i in range(self.n_sims) if self.types[i] == "NACA4"]

    def get_naca5_indices(self):
        """Get indices of NACA 5-digit simulations (for zero-shot transfer)."""
        return [i for i in range(self.n_sims) if self.types[i] == "NACA5"]


if __name__ == "__main__":
    data_dir = str(Path(__file__).resolve().parent / "data")
    try:
        ds = PLAIDDataset(data_dir, max_pairs=1000)
        print(f"\n  Dataset size: {len(ds)}")
        x, a, y, s = ds[0]
        print(f"  State shape: {x.shape}")
        print(f"  Action shape: {a.shape}")
        print(f"  Next state shape: {y.shape}")
        print(f"  Score: {s.item():.4f}")
        print(f"  NACA 4-digit: {len(ds.get_naca4_indices())}")
        print(f"  NACA 5-digit: {len(ds.get_naca5_indices())}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Run: python plaid/download_data.py first")
