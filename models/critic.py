"""
Vortaz Labs — Critic / Cost Module (LeCun Module 4)
================================================================
Implements the Cost module from LeCun's "A Path Towards Autonomous
Machine Intelligence" architecture.

The cost module computes a scalar energy for predicted latent states.
Lower energy = better predicted outcome.

In LeCun's framework:
  C(s_current, s_pred, a) -> scalar energy

Fixed from v1: Uses BOTH current and predicted latent (the difference
encodes the CHANGE caused by the action, which correlates with score).
Deeper network with residual connections for better gradient flow.
================================================================
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    LeCun Module 4: Cost / Critic.

    Takes BOTH the current latent and predicted future latent, plus the
    action, and outputs a scalar energy.

    The key insight: score_delta depends on the CHANGE in state (s_pred - s_current),
    not just s_pred alone. By providing both, the critic can learn the
    delta pattern directly.

    Energy interpretation:
      - Low energy  -> good predicted outcome (high score)
      - High energy -> bad predicted outcome (low/no score)
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 3,
                 hidden_dim: int = 256):
        super().__init__()

        # Input: current_latent + predicted_latent + delta + action
        # = latent_dim * 3 + action_dim
        input_dim = latent_dim * 3 + action_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual block 2
        self.res2 = nn.Sequential(
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

    def forward(self, s_current: torch.Tensor, s_pred: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Compute energy (cost) for a predicted state transition.

        s_current: [batch, latent_dim] — current latent state
        s_pred:    [batch, latent_dim] — predicted future latent state
        action:    [batch, action_dim] — action that led to this state

        returns: [batch, 1] — scalar energy (lower = better)
        """
        # Concatenate current, predicted, their difference, and action
        delta = s_pred - s_current
        x = torch.cat([s_current, s_pred, delta, action], dim=-1)

        h = self.input_proj(x)
        h = h + self.res1(h)    # residual
        h = h + self.res2(h)    # residual
        return self.head(h)

    def energy(self, s_current: torch.Tensor, s_pred: torch.Tensor,
               action: torch.Tensor) -> torch.Tensor:
        """Alias for forward — computes energy."""
        return self.forward(s_current, s_pred, action)


if __name__ == "__main__":
    batch = 16
    latent_dim, action_dim = 256, 3

    critic = Critic(latent_dim, action_dim)
    s_current = torch.randn(batch, latent_dim)
    s_pred = torch.randn(batch, latent_dim)
    action = torch.randn(batch, action_dim)

    energy = critic(s_current, s_pred, action)
    print(f"Energy shape: {energy.shape}")
    print(f"Energy range: [{energy.min().item():.3f}, {energy.max().item():.3f}]")

    params = sum(p.numel() for p in critic.parameters())
    print(f"Critic params: {params:,}")
