"""
Vortaz Labs — Symbolic-JEPA for PLAID AirfRANS
=================================================
Complete Symbolic-JEPA world model for computational fluid dynamics.

This is the "Symbolic-JEPA" — it predicts in coordinate/symbolic space
rather than pixel space. The key difference from standard JEPA/V-JEPA:
  - Input: symbolic state vectors (physical quantities), NOT pixels
  - Latent: causal geometry of the domain, NOT visual patches
  - Action: physical parameter changes (AoA, Reynolds)

Architecture (matches LeCun's 6-module framework):
  1. Configurator:  Reynolds number modulation
  2. Encoder:       Aggregated point cloud → 256D latent
  3. Predictor:     latent + action → predicted next latent
  4. Critic:        Energy-based physical verification
  5. Actor:         Multi-start GD for optimal configuration
  6. Memory:        Not used (steady-state RANS)

Training: Same 2-term loss as LeWorldModel (prediction + SIGReg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import SIGReg
from plaid.dataset import POOLED_DIM, ACTION_DIM


class SymbolicEncoder(nn.Module):
    """
    Symbolic Encoder: Point cloud aggregation → 256D latent.

    Unlike pixel-based JEPA encoders (ViT patches), this operates
    on pre-aggregated physical features — making it invariant to
    the specific mesh discretization used in the CFD simulation.

    This is the ONLY domain-specific module. For cross-domain transfer,
    only this module needs to be retrained.
    """

    def __init__(self, input_dim: int = POOLED_DIM,
                 hidden_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalPredictor(nn.Module):
    """
    Causal State Evolution Predictor.

    Takes current latent + action (AoA change, Re change) and predicts
    the next latent state. Constrained by RANS physics encoded in weights.

    This is the module hypothesized to be domain-agnostic: it learns
    a general "if X changes by delta, the system state evolves to Y"
    pattern that may transfer across physics domains.
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_t, a_t], dim=-1))


class PhysicalCritic(nn.Module):
    """
    Energy-Based Physical Verifier (Critic).

    Differentiates valid physical flow fields from non-physical predictions.
    Uses a contrastive margin loss during training.

    High energy = non-physical prediction (pressure discontinuities, etc)
    Low energy  = physically plausible prediction

    Target: Spearman ρ > 0.71 against OpenFOAM ground truth.
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256):
        super().__init__()

        input_dim = latent_dim * 3 + action_dim  # current + pred + delta + action

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, s_current, s_pred, action):
        delta = s_pred - s_current
        x = torch.cat([s_current, s_pred, delta, action], dim=-1)
        h = self.input_proj(x)
        h = h + self.res(h)
        return self.head(h)


class SymbolicJEPA(nn.Module):
    """
    Complete Symbolic-JEPA for PLAID AirfRANS.

    Training: L = L_prediction + λ * L_SIGReg (same as LeWorldModel)

    Key differences from pixel-JEPA:
      1. Input is symbolic (physical quantities), not pixels
      2. Latent space encodes causal geometry, not visual features
      3. Actions are physical parameter changes, not game controls
      4. Critic verifies physical plausibility, not game outcomes
    """

    def __init__(self, input_dim=POOLED_DIM, action_dim=ACTION_DIM,
                 latent_dim=256, hidden_dim=256,
                 ema_momentum=0.996, sigreg_lambda=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum

        # Modules
        self.encoder = SymbolicEncoder(input_dim, hidden_dim, latent_dim)
        self.predictor = CausalPredictor(latent_dim, action_dim, hidden_dim)
        self.critic = PhysicalCritic(latent_dim, action_dim, hidden_dim)
        self.sigreg = SIGReg(lam=sigreg_lambda)

        # Target encoder (EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        for online_p, target_p in zip(self.encoder.parameters(),
                                      self.target_encoder.parameters()):
            target_p.data = (self.ema_momentum * target_p.data +
                             (1.0 - self.ema_momentum) * online_p.data)

    def encode(self, x):
        return self.encoder(x)

    def predict(self, s_t, action):
        return self.predictor(s_t, action)

    def forward(self, x, action, next_x=None):
        """
        Forward pass.

        Training (next_x provided):
            L = L_prediction + SIGReg

        Inference (next_x=None):
            Returns predicted latent
        """
        s_t = self.encoder(x)
        s_pred = self.predictor(s_t, action)

        if next_x is None:
            return s_pred

        with torch.no_grad():
            s_target = self.target_encoder(next_x)

        # LeWM loss: prediction + SIGReg
        pred_loss = F.mse_loss(s_pred, s_target)
        sigreg_pred = self.sigreg(s_pred)
        sigreg_target = self.sigreg(s_target.detach())
        loss = pred_loss + sigreg_pred + sigreg_target

        return loss, s_pred, s_target

    def compute_critic_loss(self, s_current, s_pred, action, scores,
                             margin=1.0, oversample_ratio=2.3):
        """
        Contrastive margin loss for the physical verifier.
        With 2.3x oversampling of valid physical transitions.
        """
        energies = self.critic(s_current, s_pred, action).squeeze(-1)

        # Oversampling: positive (valid) transitions
        positive_mask = scores > scores.median()
        negative_mask = ~positive_mask

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return energies.mean()

        pos_energy = energies[positive_mask]
        neg_energy = energies[negative_mask]

        # Oversample positives
        n_over = int(len(pos_energy) * oversample_ratio)
        if n_over > len(pos_energy):
            indices = torch.randint(0, len(pos_energy), (n_over,))
            pos_energy = pos_energy[indices]

        # Contrastive margin: pos should be lower energy than neg
        n_pairs = min(len(pos_energy), len(neg_energy))
        loss = F.relu(pos_energy[:n_pairs] - neg_energy[:n_pairs] + margin)

        return loss.mean()

    def multi_step_predict(self, x, actions_list):
        """Multi-step rollout for different AoA sequences."""
        s_t = self.encoder(x)
        predictions = [s_t]
        for action in actions_list:
            s_t = self.predictor(s_t, action)
            predictions.append(s_t)
        return predictions


class SymbolicDecoder(nn.Module):
    """
    Decoder: Latent → aggregated physical features.
    Used for evaluation and interpretability only.
    """

    def __init__(self, latent_dim=256, output_dim=POOLED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


if __name__ == "__main__":
    print("=== Symbolic-JEPA for PLAID Test ===\n")

    batch = 16
    model = SymbolicJEPA()
    decoder = SymbolicDecoder()

    x = torch.randn(batch, POOLED_DIM)
    a = torch.randn(batch, ACTION_DIM)
    y = torch.randn(batch, POOLED_DIM)

    # Training forward
    loss, s_pred, s_target = model(x, a, y)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pred shape: {s_pred.shape}")
    print(f"  Target shape: {s_target.shape}")

    # Critic
    s_current = model.encoder(x)
    energy = model.critic(s_current, s_pred, a)
    print(f"  Energy shape: {energy.shape}")

    # Decoder
    decoded = decoder(s_pred)
    print(f"  Decoded shape: {decoded.shape}")

    # Multi-step
    actions = [torch.randn(batch, ACTION_DIM) for _ in range(5)]
    rollout = model.multi_step_predict(x, actions)
    print(f"  Rollout: {len(rollout)} steps")

    # Params
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    pred_p = sum(p.numel() for p in model.predictor.parameters())
    critic_p = sum(p.numel() for p in model.critic.parameters())
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Encoder:   {enc_p:,}")
    print(f"  Predictor: {pred_p:,}")
    print(f"  Critic:    {critic_p:,}")
    print(f"  Total:     {total_p:,}")
