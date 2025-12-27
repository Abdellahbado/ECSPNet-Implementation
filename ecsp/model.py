"""
ECSPNet Model Architecture.
Paper-exact implementation following Figure 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .data import TRAINING_CONFIG


class ECSPBlock(nn.Module):
    """
    Single ECSP Block: Cross-attention + Self-attention.

    Paper structure:
    1. Cross-attn: Q=tasks_emb, K/V=EP_emb
       tasks1 = LayerNorm(tasks_emb + MultiHeadAttn(Q,K,V))
    2. Self-attn: Q/K/V=tasks1
       tasks2 = LayerNorm(tasks1 + MultiHeadAttn(Q,K,V))
    """

    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Cross-attention: tasks attend to EP (electricity prices)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention: tasks attend to each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        tasks_emb: torch.Tensor,  # [B, N, d_model]
        EP_emb: torch.Tensor,  # [B, 20, d_model]
        task_mask: Optional[torch.Tensor] = None,  # [B, N] padding mask
    ) -> torch.Tensor:
        """
        Forward pass through ECSP block.

        Args:
            tasks_emb: Task embeddings [B, N, d_model]
            EP_emb: Electricity price embeddings [B, 20, d_model]
            task_mask: Boolean mask where True = ignore (padding)

        Returns:
            Updated task embeddings [B, N, d_model]
        """
        # Cross-attention: tasks attend to EP
        # Q = tasks_emb, K = V = EP_emb
        cross_out, _ = self.cross_attn(
            query=tasks_emb,
            key=EP_emb,
            value=EP_emb,
        )
        tasks1 = self.norm1(tasks_emb + cross_out)

        # Self-attention: tasks attend to each other
        # Q = K = V = tasks1
        # Apply mask for padding
        key_padding_mask = task_mask if task_mask is not None else None
        self_out, _ = self.self_attn(
            query=tasks1,
            key=tasks1,
            value=tasks1,
            key_padding_mask=key_padding_mask,
        )
        tasks2 = self.norm2(tasks1 + self_out)

        return tasks2


class ECSPNet(nn.Module):
    """
    ECSPNet: Energy-Conscious Scheduling Policy Network.

    Architecture (Paper Figure 2):
    Input → Preprocessing → [ECSP Block ×2] → Output Head

    Preprocessing:
    1. tasks_emb = Linear(tasks) → [B, N, 128]
    2. EP_emb = Linear(EP) → [B, 20, 128]
    3. objs_emb = Linear(objs) → [B, 128]
    4. w_emb = Linear(w) → [B, 128]
    5. tasks_emb += objs_emb.unsqueeze(1) + w_emb.unsqueeze(1)

    Output Head:
    logits = Linear(tasks) → [B, N, 2]
    logits[masked] = -inf
    probs = softmax(logits.view(B, -1))  # [B, 2N]
    """

    def __init__(
        self,
        d_model: int = TRAINING_CONFIG["d_model"],
        num_heads: int = TRAINING_CONFIG["num_heads"],
        num_blocks: int = TRAINING_CONFIG["num_blocks"],
        task_dim: int = 5,
        ep_horizon: int = 20,
        dropout: float = 0.0,
    ):
        """
        Initialize ECSPNet.

        Args:
            d_model: Hidden dimension (paper: 128)
            num_heads: Number of attention heads (paper: 8)
            num_blocks: Number of ECSP blocks (paper: 2)
            task_dim: Task feature dimension (5: p1, p2, p3, P_high, P_low)
            ep_horizon: Number of EP slots (paper: 20)
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Preprocessing embeddings
        self.task_embed = nn.Linear(task_dim, d_model)
        self.ep_embed = nn.Linear(1, d_model)  # Each EP slot is 0 or 1
        self.objs_embed = nn.Linear(2, d_model)  # [TWT, EEC]
        self.w_embed = nn.Linear(1, d_model)  # preference weight

        # Stack of ECSP blocks
        self.blocks = nn.ModuleList(
            [ECSPBlock(d_model, num_heads, dropout) for _ in range(num_blocks)]
        )

        # Output head: project to 2 logits per task (mode 0 and mode 1)
        self.output_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tasks: torch.Tensor,  # [B, N, 5]
        EP: torch.Tensor,  # [B, 20]
        objs: torch.Tensor,  # [B, 2]
        w: torch.Tensor,  # [B, 1]
        mask: torch.Tensor,  # [B, N] - 1=available, 0=scheduled/padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute action probabilities.

        Args:
            tasks: Task features [B, N, 5]
            EP: Electricity prices for next 20 slots [B, 20]
            objs: Current objectives [TWT, EEC] [B, 2]
            w: Preference weight [B, 1]
            mask: Availability mask [B, N] (1=available, 0=not)

        Returns:
            probs: Action probabilities [B, 2N]
            logits: Raw logits [B, 2N]
        """
        B, N, _ = tasks.shape

        # Preprocessing
        tasks_emb = self.task_embed(tasks)  # [B, N, d_model]
        EP_emb = self.ep_embed(EP.unsqueeze(-1))  # [B, 20, d_model]
        objs_emb = self.objs_embed(objs)  # [B, d_model]
        w_emb = self.w_embed(w)  # [B, d_model]

        # Add objs and w embeddings to task embeddings (broadcast)
        tasks_emb = tasks_emb + objs_emb.unsqueeze(1) + w_emb.unsqueeze(1)

        # Convert mask for attention: True = ignore (padding/scheduled)
        # Our mask: 1=available, 0=not → invert for attention
        attn_mask = mask == 0  # True where task should be ignored

        # Pass through ECSP blocks
        for block in self.blocks:
            tasks_emb = block(tasks_emb, EP_emb, attn_mask)

        # Output head: logits for each task and mode
        logits = self.output_head(tasks_emb)  # [B, N, 2]

        # Flatten to [B, 2N] actions: (task0_mode0, task0_mode1, task1_mode0, ...)
        logits = logits.view(B, -1)  # [B, 2N]

        # Apply mask: set logits for unavailable tasks to -inf
        # Expand mask to cover both modes per task
        action_mask = mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, -1)  # [B, 2N]
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        return probs, logits

    def forward_from_obs(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass from observation dictionary.

        Args:
            obs: Dictionary with keys 'tasks', 'EP', 'objs', 'w', 'mask'

        Returns:
            probs, logits
        """
        return self.forward(
            tasks=obs["tasks"],
            EP=obs["EP"],
            objs=obs["objs"],
            w=obs["w"],
            mask=obs["mask"],
        )

    def sample_action(
        self,
        probs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from probability distribution.

        Args:
            probs: Action probabilities [B, 2N]
            deterministic: If True, select argmax instead of sampling

        Returns:
            actions: Sampled actions [B]
            log_probs: Log probabilities of sampled actions [B]
        """
        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)) + 1e-8).squeeze(-1)

        return actions, log_probs

    def sample_action_truncated(
        self,
        probs: torch.Tensor,
        beta: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions using truncation sampling (for inference).

        P_trunc = ReLU(probs - beta * max(probs))
        p_prime = P_trunc / sum(P_trunc)

        Args:
            probs: Action probabilities [B, 2N]
            beta: Truncation parameter (paper: 0.9)

        Returns:
            actions: Sampled actions [B]
            log_probs: Log probabilities under original distribution [B]
        """
        # Truncation
        max_probs = probs.max(dim=-1, keepdim=True)[0]  # [B, 1]
        P_trunc = F.relu(probs - beta * max_probs)  # [B, 2N]

        # Normalize
        p_prime = P_trunc / (P_trunc.sum(dim=-1, keepdim=True) + 1e-8)

        # Sample from truncated distribution
        dist = torch.distributions.Categorical(p_prime)
        actions = dist.sample()

        # Log probs under original distribution (for consistency)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)) + 1e-8).squeeze(-1)

        return actions, log_probs

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution.

        Args:
            probs: Action probabilities [B, 2N]

        Returns:
            entropy: Mean entropy across batch [scalar]
        """
        # Avoid log(0)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        return entropy.mean()


def obs_dict_to_tensors(
    obs: Dict[str, np.ndarray],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy observation dict to torch tensors.

    Args:
        obs: Observation dictionary with numpy arrays
        device: Target device

    Returns:
        Dictionary with torch tensors
    """
    return {
        k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
        for k, v in obs.items()
    }


if __name__ == "__main__":
    # Test model
    print("Testing ECSPNet...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = ECSPNet(d_model=128, num_heads=8, num_blocks=2).to(device)

    # Print parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test forward pass
    B, N = 8, 20
    tasks = torch.randn(B, N, 5).to(device)
    EP = torch.randint(0, 2, (B, 20)).float().to(device)
    objs = torch.randn(B, 2).abs().to(device)
    w = torch.rand(B, 1).to(device)
    mask = torch.ones(B, N).to(device)
    mask[:, -5:] = 0  # Mask last 5 tasks

    probs, logits = model(tasks, EP, objs, w, mask)
    print(f"\nForward pass:")
    print(
        f"  Input shapes: tasks={tasks.shape}, EP={EP.shape}, objs={objs.shape}, w={w.shape}, mask={mask.shape}"
    )
    print(f"  Output shapes: probs={probs.shape}, logits={logits.shape}")
    print(f"  Probs sum: {probs.sum(dim=-1)}")  # Should be 1s

    # Test sampling
    actions, log_probs = model.sample_action(probs)
    print(f"\nSampling:")
    print(f"  Actions: {actions}")
    print(f"  Log probs: {log_probs}")

    # Test truncated sampling
    actions_trunc, log_probs_trunc = model.sample_action_truncated(probs, beta=0.9)
    print(f"\nTruncated sampling (beta=0.9):")
    print(f"  Actions: {actions_trunc}")

    # Test entropy
    entropy = model.compute_entropy(probs)
    print(f"\nEntropy: {entropy.item():.4f}")
