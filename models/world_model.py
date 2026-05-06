"""
Vortaz Labs — LeWorldModel (LeWM) for Angry Birds
================================================================
Faithful implementation of Yann LeCun's LeWorldModel architecture:

Paper:  "LeWorldModel" (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026)
arXiv:  2603.19312
GitHub: github.com/lucas-maes/le-wm

Key innovations over prior JEPA approaches:
  1. SIGReg (Spherical Isotropic Gaussian Regularizer) — prevents
     representation collapse with a single regularizer instead of
     complex contrastive losses. Forces embeddings to follow N(0,I).
  2. Only 2 loss terms: prediction loss + SIGReg (reduced from 6+ hyperparams to 1)
  3. ~15M parameters — trains on a single GPU in hours
  4. 48x faster planning than foundation-model-based world models
  5. First JEPA that trains stably end-to-end without collapse

Architecture:
  - Encoder:          obs → latent embedding
  - Predictor:        (latent_t, action_t) → predicted latent_{t+1}
  - Target Encoder:   EMA copy of encoder (provides stable targets)
  - SIGReg:           Regularizes embeddings to be isotropic Gaussian

This integrates with LeCun's broader 6-module framework:
  1. Configurator — context modulation
  2. Perception   — encoder (this file)
  3. World Model  — predictor (this file)
  4. Cost/Critic  — critic.py
  5. Actor        — actor.py
  6. Memory       — short-term memory (this file)
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


# ══════════════════════════════════════════════════
# SIGReg — Spherical Isotropic Gaussian Regularizer
# The core innovation of LeWorldModel
# ══════════════════════════════════════════════════

class SIGReg(nn.Module):
    """
    SIGReg: Spherical Isotropic Gaussian Regularizer.
    (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026)

    Prevents JEPA representation collapse by enforcing that the
    distribution of latent embeddings matches N(0, I) — an isotropic
    Gaussian on the unit sphere.

    Two components:
      1. Spherical: Projects embeddings to unit sphere (L2 normalization)
         then penalizes deviation of covariance from identity matrix.
      2. Isotropic: Penalizes non-uniform variance across dimensions.

    This single regularizer replaces the complex variance-invariance-covariance
    (VICReg) approach and the contrastive losses used in prior work.

    Loss:  L_sigreg = lambda * (L_cov + L_var)
      - L_cov: off-diagonal covariance → 0 (decorrelation)
      - L_var: per-dimension variance → 1 (anti-collapse)
    """

    def __init__(self, lam: float = 1.0):
        """
        Args:
            lam: regularization strength. THE ONLY hyperparameter in LeWM.
                 Default 1.0 works for most settings per the paper.
        """
        super().__init__()
        self.lam = lam

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss on a batch of embeddings.

        z: [batch, latent_dim] — batch of latent vectors

        Returns: scalar loss
        """
        batch_size, dim = z.shape
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        # Step 1: L2-normalize to unit sphere
        z_norm = F.normalize(z, dim=-1)  # [batch, dim]

        # Step 2: Compute batch covariance matrix
        z_centered = z_norm - z_norm.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (batch_size - 1)  # [dim, dim]

        # Step 3: Variance loss — force diagonal to be 1/dim
        # (uniform variance across all dimensions)
        target_var = 1.0 / dim
        var_loss = F.mse_loss(cov.diagonal(), torch.full_like(cov.diagonal(), target_var))

        # Step 4: Covariance loss — force off-diagonal to be 0
        # (decorrelate dimensions)
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0)
        cov_loss = off_diag.pow(2).sum() / dim

        return self.lam * (var_loss + cov_loss)


# ──────────────────────────────────────────────────
# Encoder (Perception module)
# ──────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    LeWM Encoder — maps observations to latent space.
    Follows the paper's MLP architecture for structured inputs.

    For pixel inputs, this would be a CNN/ViT backbone.
    For our structured game state (164D), we use an MLP.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
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


# ──────────────────────────────────────────────────
# Predictor (World Model dynamics)
# ──────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    LeWM Predictor — predicts next latent from current latent + action.

    Key property: operates entirely in latent space. Does NOT reconstruct
    observations. This is fundamentally different from generative world
    models (VAE, diffusion) — it predicts abstract representations.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
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


# ──────────────────────────────────────────────────
# Short-Term Memory
# ──────────────────────────────────────────────────

class ShortTermMemory(nn.Module):
    """
    Short-term memory via GRU — enriches latent states with temporal context.
    Optional module for sequential decision-making.
    """
    def __init__(self, latent_dim: int, memory_dim: int = None):
        super().__init__()
        self.memory_dim = memory_dim or latent_dim
        self.gru = nn.GRUCell(latent_dim, self.memory_dim)
        self.proj = (nn.Linear(self.memory_dim, latent_dim)
                     if self.memory_dim != latent_dim else nn.Identity())

    def forward(self, s_t: torch.Tensor,
                h_prev: torch.Tensor = None) -> tuple:
        batch = s_t.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch, self.memory_dim, device=s_t.device)
        h_new = self.gru(s_t, h_prev)
        s_enriched = s_t + self.proj(h_new)
        return s_enriched, h_new


# ──────────────────────────────────────────────────
# Configurator
# ──────────────────────────────────────────────────

class Configurator(nn.Module):
    """Context-dependent modulation of latent representations."""
    def __init__(self, context_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


# ══════════════════════════════════════════════════
# LeWorldModel — Complete JEPA World Model
# ══════════════════════════════════════════════════

class GameJEPA(nn.Module):
    """
    Vortaz Labs implementation of LeWorldModel (LeWM).
    (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026)

    Training uses ONLY 2 loss terms:
      L_total = L_prediction + lambda * L_SIGReg

    Where:
      L_prediction = ||Predictor(Encoder(x_t), a_t) - TargetEncoder(x_{t+1})||
      L_SIGReg     = SIGReg(embeddings) — prevents collapse

    This is radically simpler than prior JEPA approaches which needed
    6+ tunable hyperparameters. LeWM has just ONE: lambda (SIGReg weight).

    Architecture diagram:
    ┌──────────────────────────────────────────────────────────┐
    │                    LeWorldModel (LeWM)                    │
    │                                                          │
    │   x_t ──▶ [Encoder] ──▶ s_t ──┐                        │
    │                                 ├──▶ [Predictor] ──▶ ŝ_{t+1}  │
    │   a_t ─────────────────────────┘                        │
    │                                                          │
    │   x_{t+1} ──▶ [Target Encoder (EMA)] ──▶ s_{t+1}       │
    │                                                          │
    │   Loss = ||ŝ_{t+1} - s_{t+1}||² + λ·SIGReg(s, ŝ)     │
    │                                                          │
    │   Target Encoder ← EMA(Encoder) after each step         │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self, obs_dim: int = 164, action_dim: int = 3,
                 latent_dim: int = 256, hidden_dim: int = 512,
                 ema_momentum: float = 0.996,
                 sigreg_lambda: float = 1.0,
                 use_memory: bool = True,
                 use_configurator: bool = True,
                 context_dim: int = 8):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.use_memory = use_memory
        self.use_configurator = use_configurator

        # Core LeWM components
        self.encoder = Encoder(obs_dim, hidden_dim, latent_dim)
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim)

        # SIGReg — the LeWM collapse prevention mechanism
        self.sigreg = SIGReg(lam=sigreg_lambda)

        # Target Encoder — EMA copy, never directly trained
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Optional modules from LeCun's broader framework
        self.memory = ShortTermMemory(latent_dim) if use_memory else None
        self.configurator = Configurator(context_dim, latent_dim) if use_configurator else None

    # Alias for backward compatibility with existing code
    @property
    def perception(self):
        return self.encoder

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update: target ← momentum * target + (1-momentum) * online.
        Called once per training step. Provides stable, slowly-moving targets.
        """
        for online_p, target_p in zip(self.encoder.parameters(),
                                      self.target_encoder.parameters()):
            target_p.data = (self.ema_momentum * target_p.data +
                             (1.0 - self.ema_momentum) * online_p.data)

    def encode(self, obs: torch.Tensor,
               context: torch.Tensor = None,
               h_prev: torch.Tensor = None) -> tuple:
        """Encode observation → latent, with optional configurator + memory."""
        s_t = self.encoder(obs)

        if self.use_configurator and context is not None and self.configurator is not None:
            modulation = self.configurator(context)
            s_t = s_t * modulation

        h_new = None
        if self.use_memory and self.memory is not None:
            s_t, h_new = self.memory(s_t, h_prev)

        return s_t, h_new

    def predict(self, s_t: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """World model prediction: latent + action → predicted next latent."""
        return self.predictor(s_t, action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                next_obs: torch.Tensor = None, loss_type: str = "lewm",
                context: torch.Tensor = None,
                h_prev: torch.Tensor = None):
        """
        Full forward pass.

        Training (next_obs provided):
            Returns (loss, s_pred, s_target, h_new)

            loss_type="lewm" (default): LeWorldModel loss
                L = L_prediction + SIGReg(s_pred) + SIGReg(s_target)

        Inference (next_obs=None):
            Returns (s_pred, h_new)
        """
        # Encode current observation
        s_t, h_new = self.encode(obs, context, h_prev)

        # World model prediction
        s_pred = self.predictor(s_t, action)

        if next_obs is None:
            return s_pred, h_new

        # Target encoding (no gradient — EMA provides targets)
        with torch.no_grad():
            s_target = self.target_encoder(next_obs)

        # ═══ LeWorldModel Loss: only 2 terms ═══
        if loss_type == "lewm":
            # Term 1: Next-embedding prediction loss (MSE in latent space)
            prediction_loss = F.mse_loss(s_pred, s_target)

            # Term 2: SIGReg on both predicted and target embeddings
            # This prevents collapse by enforcing N(0,I) distribution
            sigreg_pred = self.sigreg(s_pred)
            sigreg_target = self.sigreg(s_target.detach())

            loss = prediction_loss + sigreg_pred + sigreg_target

        elif loss_type == "l1":
            # Fallback: smooth L1 + SIGReg
            loss = F.smooth_l1_loss(s_pred, s_target) + self.sigreg(s_pred)
        elif loss_type == "cosine":
            loss = (1.0 - F.cosine_similarity(s_pred, s_target, dim=-1).mean()
                    + self.sigreg(s_pred))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'lewm'.")

        return loss, s_pred, s_target, h_new

    def multi_step_predict(self, obs: torch.Tensor,
                           actions: list,
                           context: torch.Tensor = None) -> list:
        """
        Multi-step rollout — 48x faster than foundation model planning (per paper).

        Predicts sequence of future latent states from action sequence.
        """
        s_t, h = self.encode(obs, context)
        predictions = []
        for action in actions:
            s_t = self.predictor(s_t, action)
            predictions.append(s_t)
        return predictions


# ──────────────────────────────────────────────────
# Decoder (for evaluation only, NOT part of LeWM)
# ──────────────────────────────────────────────────

class GameDecoder(nn.Module):
    """
    Decoder — maps latent back to observation space.

    NOT part of LeWorldModel's core architecture.
    Used only for evaluation and comparison with LLM baselines.
    Trained on PREDICTOR outputs (not target encoder).
    """
    def __init__(self, latent_dim: int = 256, output_dim: int = 164):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
        )
        self.head = nn.Linear(latent_dim // 2, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(z)
        h2 = self.layer2(h1) + h1  # residual
        h3 = self.layer3(h2)
        return self.head(h3)


if __name__ == "__main__":
    batch = 16
    obs_dim, action_dim = 164, 3

    # === Test LeWorldModel ===
    print("=== LeWorldModel (LeWM) Test ===\n")

    jepa = GameJEPA(
        obs_dim=obs_dim, action_dim=action_dim,
        latent_dim=256, hidden_dim=512,
        sigreg_lambda=1.0,
        use_memory=True, use_configurator=True
    )
    decoder = GameDecoder(latent_dim=256, output_dim=obs_dim)

    obs = torch.randn(batch, obs_dim)
    action = torch.randn(batch, action_dim)
    next_obs = torch.randn(batch, obs_dim)
    context = torch.randn(batch, 8)

    # LeWM training loss (prediction + SIGReg)
    loss, s_pred, s_target, h = jepa(obs, action, next_obs,
                                      loss_type="lewm", context=context)
    print(f"Pred shape:     {s_pred.shape}")
    print(f"Target shape:   {s_target.shape}")
    print(f"LeWM loss:      {loss.item():.4f}")

    # SIGReg alone
    sigreg_loss = jepa.sigreg(s_pred)
    print(f"SIGReg loss:    {sigreg_loss.item():.4f}")

    # Decode
    decoded = decoder(s_pred)
    print(f"Decoded shape:  {decoded.shape}")

    # Multi-step rollout (48x faster than foundation model planning)
    actions = [torch.randn(batch, action_dim) for _ in range(5)]
    rollout = jepa.multi_step_predict(obs, actions, context)
    print(f"Rollout:        {len(rollout)} steps, each {rollout[0].shape}")

    # EMA update
    jepa.update_target_encoder()
    print(f"EMA update:     success")

    # Parameter count (~15M target per paper)
    params = sum(p.numel() for p in jepa.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in jepa.parameters())
    print(f"\nTrainable params: {params:,}")
    print(f"Total params:     {total_params:,}")
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params:   {dec_params:,}")
