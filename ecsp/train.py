"""
Training module - Algorithm 1 from paper.
Paper-exact implementation.

VERSION: 2.0-GPU - Full GPU acceleration for training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import json
from datetime import datetime

from .data import (
    generate_batch,
    compute_total_processing_time,
    TRAINING_CONFIG,
    ENV_CONFIG,
    BENCHMARK_SCALES,
)
from .env import GPUBatchECSPEnv
from .model import ECSPNet

# Version identifier
TRAIN_VERSION = "2.0-GPU"
print(f"[Trainer v{TRAIN_VERSION}] Loading GPU-accelerated training module...")

__all__ = ["Trainer", "train_model", "TRAIN_VERSION"]


class Trainer:
    """
    ECSP Training using REINFORCE with baseline.

    Paper Algorithm 1:
    1. Generate batches of instances
    2. Sample trajectories using policy
    3. Compute rewards R = -max(w*TWT, (1-w)*EEC)
    4. Compute baseline using 10 w-bins
    5. Policy gradient with entropy regularization
    """

    def __init__(
        self,
        N: int = 20,
        d_model: int = TRAINING_CONFIG["d_model"],
        num_heads: int = TRAINING_CONFIG["num_heads"],
        num_blocks: int = TRAINING_CONFIG["num_blocks"],
        batch_size: int = TRAINING_CONFIG["batch_size"],
        batches_per_epoch: int = TRAINING_CONFIG["batches_per_epoch"],
        num_epochs: int = TRAINING_CONFIG["epochs"],
        initial_lr: float = TRAINING_CONFIG["initial_lr"],
        lr_decay_epochs: List[int] = TRAINING_CONFIG["lr_decay_epochs"],
        lr_decay_factor: float = TRAINING_CONFIG["lr_decay_factor"],
        entropy_coef: float = TRAINING_CONFIG["entropy_coef"],
        num_w_bins: int = ENV_CONFIG["num_w_bins"],
        device: str = "cuda",
        save_dir: str = "checkpoints",
    ):
        """
        Initialize trainer.

        Args:
            N: Number of tasks per instance
            d_model: Model hidden dimension
            num_heads: Number of attention heads
            num_blocks: Number of ECSP blocks
            batch_size: Training batch size (paper: 2048)
            batches_per_epoch: Number of batches per epoch (paper: 50)
            num_epochs: Total training epochs (paper: 3000)
            initial_lr: Initial learning rate (paper: 1e-3)
            lr_decay_epochs: Epochs to decay LR (paper: [1000, 2000])
            lr_decay_factor: LR decay factor (paper: 0.1)
            entropy_coef: Entropy coefficient alpha (paper: 0.1)
            num_w_bins: Number of bins for baseline (paper: 10)
            device: Training device
            save_dir: Directory to save checkpoints
        """
        self.N = N
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_factor = lr_decay_factor
        self.entropy_coef = entropy_coef
        self.num_w_bins = num_w_bins
        self.save_dir = save_dir

        # Setup device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Trainer v{TRAIN_VERSION}] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Trainer v{TRAIN_VERSION}] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer v{TRAIN_VERSION}] CUDA version: {torch.version.cuda}")

        # Create model
        self.model = ECSPNet(
            d_model=d_model,
            num_heads=num_heads,
            num_blocks=num_blocks,
        ).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr)

        # Create GPU-native environment
        self.env = GPUBatchECSPEnv(N=N, batch_size=batch_size, device=self.device)

        # Training history
        self.history = {
            "epoch": [],
            "loss": [],
            "policy_loss": [],
            "entropy": [],
            "mean_reward": [],
            "mean_twt": [],
            "mean_eec": [],
            "lr": [],
        }

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def compute_baseline(
        self,
        rewards: torch.Tensor,  # [batch] GPU tensor
        ws: torch.Tensor,  # [batch] GPU tensor
        T_pts: torch.Tensor,  # [batch] GPU tensor
    ) -> torch.Tensor:
        """
        Compute baseline using 10 w-bins. All on GPU.

        Paper Algorithm 1:
        - Group instances into bins based on w
        - For each instance i in bin A_k:
          b[i] = T_pt[i] * mean_{j in A_k}(R[j] / T_pt[j])

        Args:
            rewards: Rewards for each instance [batch] (GPU)
            ws: Preference weights [batch] (GPU)
            T_pts: Total processing times [batch] (GPU)

        Returns:
            baselines: Baseline values [batch] (GPU)
        """
        baselines = torch.zeros_like(rewards)

        # Define bin edges
        bin_edges = torch.linspace(0, 1, self.num_w_bins + 1, device=self.device)

        for k in range(self.num_w_bins):
            # Find instances in this bin
            if k == self.num_w_bins - 1:
                bin_mask = (ws >= bin_edges[k]) & (ws <= bin_edges[k + 1])
            else:
                bin_mask = (ws >= bin_edges[k]) & (ws < bin_edges[k + 1])

            if not bin_mask.any():
                continue

            # Compute mean(R/T_pt) for this bin
            R_over_Tpt = rewards[bin_mask] / (T_pts[bin_mask] + 1e-8)
            mean_R_over_Tpt = R_over_Tpt.mean()

            # Baseline for each instance in bin
            baselines[bin_mask] = T_pts[bin_mask] * mean_R_over_Tpt

        return baselines

    def rollout_batch(
        self,
        tasks_batch: torch.Tensor,  # [batch, N, 5] GPU tensor
        ws: torch.Tensor,  # [batch] GPU tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rollout trajectories entirely on GPU.

        Args:
            tasks_batch: Batch of task instances [batch, N, 5] (GPU)
            ws: Preference weights [batch] (GPU)

        Returns:
            twts: Total wait times [batch] (GPU)
            eecs: Energy costs [batch] (GPU)
            total_log_probs: Sum of log probs for each trajectory [batch] (GPU)
            total_entropy: Mean entropy across all steps [scalar] (GPU)
        """
        self.model.train()

        # Reset environment with GPU tensors
        obs = self.env.reset(tasks=tasks_batch, ws=ws)

        total_log_probs = torch.zeros(self.batch_size, device=self.device)
        entropies = []

        # Rollout until all done
        for step in range(self.N):
            # Forward pass - obs is already GPU tensors
            probs, logits = self.model(
                obs["tasks"], obs["EP"], obs["objs"], obs["w"], obs["mask"]
            )

            # Sample actions (stays on GPU)
            actions, log_probs = self.model.sample_action(probs)

            # Compute entropy
            entropy = self.model.compute_entropy(probs)
            entropies.append(entropy)

            # Accumulate log probs
            total_log_probs = total_log_probs + log_probs

            # Environment step (actions already on GPU)
            obs, rewards, dones, info = self.env.step(actions)

            if dones.all():
                break

        # Get final metrics (GPU tensors)
        twts, eecs = self.env.get_final_metrics()

        # Mean entropy across steps
        mean_entropy = torch.stack(entropies).mean()

        return twts, eecs, total_log_probs, mean_entropy

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch. All computation on GPU.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        epoch_losses = []
        epoch_policy_losses = []
        epoch_entropies = []
        epoch_rewards = []
        epoch_twts = []
        epoch_eecs = []

        for batch_idx in range(self.batches_per_epoch):
            # Generate new batch of instances (50 × 2048 per epoch)
            # Data generation on CPU, then move to GPU once
            tasks_batch_np = generate_batch(self.N, self.batch_size)
            tasks_batch = torch.from_numpy(tasks_batch_np).to(self.device)

            # Sample preference weights uniformly (on GPU)
            ws = torch.rand(self.batch_size, device=self.device) * 0.98 + 0.01

            # Rollout trajectories (all on GPU)
            twts, eecs, total_log_probs, entropy = self.rollout_batch(tasks_batch, ws)

            # Compute rewards on GPU: R = -max(w*TWT, (1-w)*EEC)
            rewards = -torch.maximum(ws * twts, (1 - ws) * eecs)

            # Compute T_pt on GPU (sum of p2)
            T_pts = tasks_batch[:, :, 1].sum(dim=1)  # [batch]

            # Compute baseline (all on GPU)
            baselines = self.compute_baseline(rewards, ws, T_pts)

            # Policy gradient loss (already on GPU)
            advantages = rewards - baselines
            policy_loss = -(advantages.detach() * total_log_probs).mean()

            # Entropy loss (maximize entropy, so negative)
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            loss = policy_loss + entropy_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional, for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record metrics (move to CPU only for logging)
            epoch_losses.append(loss.item())
            epoch_policy_losses.append(policy_loss.item())
            epoch_entropies.append(entropy.item())
            epoch_rewards.append(rewards.mean().item())
            epoch_twts.append(twts.mean().item())
            epoch_eecs.append(eecs.mean().item())

        return {
            "loss": np.mean(epoch_losses),
            "policy_loss": np.mean(epoch_policy_losses),
            "entropy": np.mean(epoch_entropies),
            "mean_reward": np.mean(epoch_rewards),
            "mean_twt": np.mean(epoch_twts),
            "mean_eec": np.mean(epoch_eecs),
        }

        return {
            "loss": np.mean(epoch_losses),
            "policy_loss": np.mean(epoch_policy_losses),
            "entropy": np.mean(epoch_entropies),
            "mean_reward": np.mean(epoch_rewards),
            "mean_twt": np.mean(epoch_twts),
            "mean_eec": np.mean(epoch_eecs),
        }

    def adjust_learning_rate(self, epoch: int):
        """Adjust learning rate according to schedule."""
        lr = self.initial_lr
        for decay_epoch in self.lr_decay_epochs:
            if epoch >= decay_epoch:
                lr *= self.lr_decay_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def train(
        self,
        resume_from: Optional[str] = None,
        save_every: int = 100,
    ):
        """
        Full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
            save_every: Save checkpoint every N epochs
        """
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed from epoch {start_epoch}")

        print("=" * 60)
        print(f"[Trainer v{TRAIN_VERSION}] GPU-ACCELERATED TRAINING")
        print("=" * 60)
        print(f"Starting training from epoch {start_epoch}")
        print(
            f"Configuration: N={self.N}, batch_size={self.batch_size}, "
            f"epochs={self.num_epochs}"
        )
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        print("=" * 60)

        progress_bar = tqdm(range(start_epoch, self.num_epochs), desc="Training")

        for epoch in progress_bar:
            # Adjust learning rate
            current_lr = self.adjust_learning_rate(epoch)

            # Train epoch
            metrics = self.train_epoch(epoch)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["loss"].append(metrics["loss"])
            self.history["policy_loss"].append(metrics["policy_loss"])
            self.history["entropy"].append(metrics["entropy"])
            self.history["mean_reward"].append(metrics["mean_reward"])
            self.history["mean_twt"].append(metrics["mean_twt"])
            self.history["mean_eec"].append(metrics["mean_eec"])
            self.history["lr"].append(current_lr)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.4f}",
                    "twt": f"{metrics['mean_twt']:.4f}",
                    "eec": f"{metrics['mean_eec']:.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)

        # Save final checkpoint
        self.save_checkpoint(self.num_epochs, is_final=True)
        print("Training complete!")

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save training checkpoint."""
        filename = (
            f"ecspnet_N{self.N}_epoch{epoch}.pt"
            if not is_final
            else f"ecspnet_N{self.N}_final.pt"
        )
        path = os.path.join(self.save_dir, filename)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": {
                    "N": self.N,
                    "d_model": self.model.d_model,
                    "num_heads": self.model.num_heads,
                    "num_blocks": self.model.num_blocks,
                    "batch_size": self.batch_size,
                    "batches_per_epoch": self.batches_per_epoch,
                    "num_epochs": self.num_epochs,
                    "initial_lr": self.initial_lr,
                    "lr_decay_epochs": self.lr_decay_epochs,
                    "lr_decay_factor": self.lr_decay_factor,
                    "entropy_coef": self.entropy_coef,
                },
            },
            path,
        )

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def save_history(self, filename: str = None):
        """Save training history to JSON."""
        if filename is None:
            filename = (
                f"history_N{self.N}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f)

        print(f"Saved history to {path}")


def train_model(
    N: int = 20,
    num_epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    device: str = "cuda",
    save_dir: str = "checkpoints",
    resume_from: str = None,
):
    """
    Convenience function to train a model.

    Args:
        N: Number of tasks
        num_epochs: Training epochs
        batch_size: Batch size
        device: Device to train on
        save_dir: Directory for checkpoints
        resume_from: Path to resume from
    """
    trainer = Trainer(
        N=N,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_dir=save_dir,
    )

    trainer.train(resume_from=resume_from)
    trainer.save_history()

    return trainer


if __name__ == "__main__":
    # Test training
    print("Testing training module...")

    # Small test run
    trainer = Trainer(
        N=10,
        batch_size=32,
        batches_per_epoch=2,
        num_epochs=5,
        device="cpu",
        save_dir="test_checkpoints",
    )

    # Test single batch rollout (use GPU tensors)
    tasks_np = generate_batch(10, 32)
    tasks = torch.from_numpy(tasks_np).to(trainer.device)
    ws = torch.rand(32, device=trainer.device) * 0.98 + 0.01

    twts, eecs, log_probs, entropy = trainer.rollout_batch(tasks, ws)
    print(f"\nRollout test:")
    print(f"  TWTs: mean={twts.mean().item():.4f}, std={twts.std().item():.4f}")
    print(f"  EECs: mean={eecs.mean().item():.4f}, std={eecs.std().item():.4f}")
    print(f"  Log probs: mean={log_probs.mean().item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")

    # Test full training loop
    print("\nRunning short training test...")
    trainer.train()
    print("Training test complete!")
