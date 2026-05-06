"""
Vortaz Labs — Grasping Domain Encoder + Transfer-Ready JEPA
==============================================================
New Encoder for the grasping domain that maps 32D grasping state
to the SAME 256D latent space used by the Science Birds JEPA.

For transfer:
  - Predictor (256D + 3D → 256D): FROZEN from Science Birds
  - Critic (256D → scalar): FROZEN from Science Birds
  - Encoder (32D → 256D): NEW, trained on grasping data

If the Predictor/Critic capture domain-agnostic causal structure,
then only retraining the Encoder should yield good performance.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_model import GameJEPA, Encoder, Predictor, SIGReg
from domains.grasp_simulator import OBS_DIM as GRASP_OBS_DIM, ACTION_DIM as GRASP_ACTION_DIM


class GraspEncoder(nn.Module):
    """
    Encoder for the 2D grasping domain.
    Maps 32D grasping state → 256D latent (same as Science Birds).
    """
    def __init__(self, obs_dim: int = GRASP_OBS_DIM,
                 hidden_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LatentAdapter(nn.Module):
    """
    Lightweight projection layer between grasping encoder and frozen predictor.

    The frozen predictor expects latents from the Science Birds encoder, which
    occupy a specific region of R^256. This adapter learns to map the grasping
    encoder's outputs into that same distribution, bridging the domain gap.

    Architecture: LayerNorm → Linear → GELU → Linear + residual
    (keeps it lightweight so most of the representation comes from the encoder)
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # Initialize near identity (residual starts as passthrough)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.proj(self.norm(z))


class GraspJEPA(nn.Module):
    """
    JEPA World Model for grasping domain.

    Two operating modes:
      1. FROM SCRATCH: all modules freshly initialized
      2. TRANSFER: Predictor + SIGReg loaded from Science Birds checkpoint,
                   only the Encoder is newly trained

    This tests the core hypothesis: is the Predictor domain-agnostic?
    """

    def __init__(self, obs_dim=GRASP_OBS_DIM, action_dim=GRASP_ACTION_DIM,
                 latent_dim=256, hidden_dim=256,
                 ema_momentum=0.996, sigreg_lambda=1.0,
                 use_adapter=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.use_adapter = use_adapter

        # Domain-specific encoder
        self.encoder = GraspEncoder(obs_dim, hidden_dim, latent_dim)

        # Latent adapter (only used in transfer mode)
        self.adapter = LatentAdapter(latent_dim) if use_adapter else nn.Identity()

        # Predictor (potentially transferred from Science Birds)
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim)

        # SIGReg
        self.sigreg = SIGReg(lam=sigreg_lambda)

        # Target encoder (EMA copy)
        import copy
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_adapter = copy.deepcopy(self.adapter) if use_adapter else nn.Identity()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        if use_adapter:
            for p in self.target_adapter.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        for online_p, target_p in zip(self.encoder.parameters(),
                                      self.target_encoder.parameters()):
            target_p.data = (self.ema_momentum * target_p.data +
                             (1.0 - self.ema_momentum) * online_p.data)
        if self.use_adapter:
            for online_p, target_p in zip(self.adapter.parameters(),
                                          self.target_adapter.parameters()):
                target_p.data = (self.ema_momentum * target_p.data +
                                 (1.0 - self.ema_momentum) * online_p.data)

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, s_t, action):
        return self.predictor(s_t, action)

    def forward(self, obs, action, next_obs=None):
        s_t = self.adapter(self.encoder(obs))
        s_pred = self.predictor(s_t, action)

        if next_obs is None:
            return s_pred

        with torch.no_grad():
            if self.use_adapter:
                s_target = self.target_adapter(self.target_encoder(next_obs))
            else:
                s_target = self.target_encoder(next_obs)

        import torch.nn.functional as F
        pred_loss = F.mse_loss(s_pred, s_target)
        sigreg_loss = self.sigreg(s_pred) + self.sigreg(s_target.detach())
        loss = pred_loss + sigreg_loss

        return loss, s_pred, s_target

    @classmethod
    def from_transfer(cls, science_birds_ckpt: str, device="cpu",
                      unfreeze_last_predictor_layer=True):
        """
        Create a GraspJEPA with Predictor weights transferred
        from a Science Birds checkpoint.

        Architecture fixes for positive transfer:
          1. LatentAdapter bridges distribution gap between domains
          2. Last predictor layer unfrozen for mild adaptation
          3. Encoder + Adapter are fully trainable
        """
        print(f"  Loading Science Birds JEPA for transfer...")

        # Load source model
        source = GameJEPA(obs_dim=164, action_dim=3, latent_dim=256,
                          hidden_dim=512, use_memory=False,
                          use_configurator=False).to(device)
        ckpt = torch.load(science_birds_ckpt, map_location=device)
        source.load_state_dict(ckpt['model_state_dict'])
        print(f"  Source model loaded from {science_birds_ckpt}")

        # Create new grasping model WITH adapter
        grasp_model_transfer = cls(obs_dim=GRASP_OBS_DIM, action_dim=GRASP_ACTION_DIM,
                                   latent_dim=256, hidden_dim=512,
                                   use_adapter=True).to(device)

        # Copy predictor weights directly
        grasp_model_transfer.predictor.load_state_dict(source.predictor.state_dict())
        pred_params = sum(p.numel() for p in source.predictor.parameters())
        print(f"  Predictor transferred ({pred_params:,} params)")

        # Freeze predictor (except optionally last layer)
        for name, p in grasp_model_transfer.predictor.named_parameters():
            p.requires_grad = False

        unfrozen_pred = 0
        if unfreeze_last_predictor_layer:
            # Unfreeze last linear layer for mild domain adaptation
            predictor_layers = list(grasp_model_transfer.predictor.net.children())
            for layer in predictor_layers[-1:]:
                if hasattr(layer, 'parameters'):
                    for p in layer.parameters():
                        p.requires_grad = True
                        unfrozen_pred += p.numel()
            print(f"  Predictor: {pred_params - unfrozen_pred:,} frozen + "
                  f"{unfrozen_pred:,} unfrozen (last layer)")
        else:
            print(f"  Predictor FULLY FROZEN")

        adapter_params = sum(p.numel() for p in grasp_model_transfer.adapter.parameters())
        print(f"  LatentAdapter: {adapter_params:,} params (bridges domain gap)")

        trainable = sum(p.numel() for p in grasp_model_transfer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in grasp_model_transfer.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        print(f"  Encoder + Adapter + predictor-last-layer")

        return grasp_model_transfer


if __name__ == "__main__":
    print("=== Grasping JEPA Test ===\n")

    # Test from scratch
    model = GraspJEPA()
    obs = torch.randn(16, GRASP_OBS_DIM)
    action = torch.randn(16, GRASP_ACTION_DIM)
    next_obs = torch.randn(16, GRASP_OBS_DIM)

    loss, s_pred, s_target = model(obs, action, next_obs)
    print(f"From scratch:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pred shape: {s_pred.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test transfer (if checkpoint exists)
    ckpt_path = Path("checkpoints/game_jepa_best_v2.pth")
    if ckpt_path.exists():
        print(f"\nTransfer from Science Birds:")
        transfer_model = GraspJEPA.from_transfer(str(ckpt_path))
        loss_t, s_pred_t, _ = transfer_model(obs, action, next_obs)
        print(f"  Loss: {loss_t.item():.4f}")
    else:
        print(f"\n  No Science Birds checkpoint found for transfer test")
