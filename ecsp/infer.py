"""
Inference module - Algorithm 2 from paper.
Paper-exact implementation with Pareto front computation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .data import (
    generate_instance,
    INFERENCE_CONFIG,
    BENCHMARK_SCALES,
)
from .env import ECSPEnv
from .model import ECSPNet, obs_dict_to_tensors


def _make_weight_sweep(
    B: int,
    mode: str = "biased",
    exponent: float = 0.4,
    w_min: float = 0.01,
    w_max: float = 0.99,
) -> np.ndarray:
    """Generate B weights in (0,1), optionally biased toward high w."""
    if B <= 0:
        return np.empty((0,), dtype=np.float64)

    if mode == "uniform":
        # Paper sweep: w = i / (B + 1), i=1..B
        return np.array([i / (B + 1) for i in range(1, B + 1)], dtype=np.float64)

    if mode == "biased":
        u = np.linspace(0.001, 0.999, B, dtype=np.float64)
        w = u ** float(exponent)
        w = (w - w.min()) / (w.max() - w.min() + 1e-12)
        w = w * (w_max - w_min) + w_min
        return w

    raise ValueError(
        f"Unknown weight sampling mode: {mode}. Use 'uniform' or 'biased'."
    )


@dataclass
class Solution:
    """A single solution with its objectives."""

    twt: float
    eec: float
    w: float
    trajectory: Optional[List[int]] = None

    def dominates(self, other: "Solution") -> bool:
        """Check if this solution dominates another (minimization)."""
        return (
            self.twt <= other.twt
            and self.eec <= other.eec
            and (self.twt < other.twt or self.eec < other.eec)
        )


def compute_pareto_front(solutions: List[Solution]) -> List[Solution]:
    """
    Filter solutions to get Pareto front (non-dominated solutions).

    Args:
        solutions: List of all solutions

    Returns:
        List of non-dominated solutions
    """
    pareto_front = []

    for candidate in solutions:
        dominated = False
        for other in solutions:
            if other.dominates(candidate):
                dominated = True
                break

        if not dominated:
            # Check if not already in pareto front (avoid duplicates)
            is_duplicate = False
            for pf_sol in pareto_front:
                if (
                    abs(pf_sol.twt - candidate.twt) < 1e-6
                    and abs(pf_sol.eec - candidate.eec) < 1e-6
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                pareto_front.append(candidate)

    # Sort by TWT (ascending)
    pareto_front.sort(key=lambda s: s.twt)

    return pareto_front


def compute_hypervolume(
    pareto_front: List[Solution],
    reference_point: Tuple[float, float] = None,
) -> float:
    """
    Compute hypervolume indicator for a Pareto front.

    Args:
        pareto_front: List of Pareto-optimal solutions
        reference_point: Reference point for HV computation (worst case)

    Returns:
        Hypervolume value
    """
    if len(pareto_front) == 0:
        return 0.0

    # Default reference: max values + margin
    if reference_point is None:
        max_twt = max(s.twt for s in pareto_front) * 1.1
        max_eec = max(s.eec for s in pareto_front) * 1.1
        reference_point = (max_twt, max_eec)

    # Sort by TWT ascending
    sorted_front = sorted(pareto_front, key=lambda s: s.twt)

    # Compute HV using inclusion-exclusion
    hv = 0.0
    prev_eec = reference_point[1]

    for sol in sorted_front:
        if sol.twt < reference_point[0] and sol.eec < reference_point[1]:
            width = reference_point[0] - sol.twt
            height = prev_eec - sol.eec
            hv += width * height
            prev_eec = sol.eec

    return hv


class Inferencer:
    """
    ECSP Inference using trained model.

    Paper Algorithm 2:
    For i = 1 to B (1000):
        w = i / (B + 1)
        Sample trajectory using truncation sampling (β=0.9)
        Record (TWT, EEC)
    Return Pareto front of all solutions
    """

    def __init__(
        self,
        model: ECSPNet,
        device: torch.device = torch.device("cpu"),
        B: int = INFERENCE_CONFIG["num_solutions"],
        beta: float = INFERENCE_CONFIG["beta"],
        w_sampling: str = "biased",
        w_exponent: float = 0.4,
    ):
        """
        Initialize inferencer.

        Args:
            model: Trained ECSPNet model
            device: Inference device
            B: Number of solutions to generate (paper: 1000)
            beta: Truncation parameter (paper: 0.9)
        """
        self.model = model
        self.device = device
        self.B = B
        self.beta = beta
        self.w_sampling = w_sampling
        self.w_exponent = w_exponent

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def solve_instance(
        self,
        tasks: np.ndarray,
        return_all: bool = False,
        verbose: bool = False,
    ) -> Tuple[List[Solution], List[Solution]]:
        """
        Solve a single instance by generating B solutions.

        Args:
            tasks: Task features [N, 5]
            return_all: If True, return all solutions (not just Pareto front)
            verbose: Print progress

        Returns:
            pareto_front: List of Pareto-optimal solutions
            all_solutions: All generated solutions (if return_all=True)
        """
        N = len(tasks)
        env = ECSPEnv(N=N)

        all_solutions = []

        weights = _make_weight_sweep(
            self.B,
            mode=self.w_sampling,
            exponent=self.w_exponent,
        )
        iterator = weights
        if verbose:
            iterator = tqdm(weights, desc="Generating solutions")

        for w in iterator:

            # Rollout with truncation sampling
            twt, eec, trajectory = self._rollout_single(env, tasks, float(w))

            all_solutions.append(
                Solution(
                    twt=twt,
                    eec=eec,
                    w=float(w),
                    trajectory=trajectory if return_all else None,
                )
            )

        # Compute Pareto front
        pareto_front = compute_pareto_front(all_solutions)

        if return_all:
            return pareto_front, all_solutions
        else:
            return pareto_front, []

    def _rollout_single(
        self,
        env: ECSPEnv,
        tasks: np.ndarray,
        w: float,
    ) -> Tuple[float, float, List[int]]:
        """
        Rollout a single trajectory with truncation sampling.

        Args:
            env: ECSP environment
            tasks: Task features [N, 5]
            w: Preference weight

        Returns:
            twt, eec, trajectory
        """
        obs, _ = env.reset(options={"tasks": tasks, "w": w})
        obs_tensors = obs_dict_to_tensors(obs, self.device)

        # Add batch dimension
        for k, v in obs_tensors.items():
            obs_tensors[k] = v.unsqueeze(0)

        trajectory = []

        while True:
            # Forward pass
            probs, logits = self.model.forward_from_obs(obs_tensors)

            # Truncation sampling
            actions, _ = self.model.sample_action_truncated(probs, self.beta)
            action = actions.item()

            trajectory.append(action)

            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            obs_tensors = obs_dict_to_tensors(obs, self.device)
            for k, v in obs_tensors.items():
                obs_tensors[k] = v.unsqueeze(0)

        twt, eec = env.get_final_metrics()

        return twt, eec, trajectory

    @torch.no_grad()
    def solve_batch(
        self,
        instances: List[np.ndarray],
        verbose: bool = True,
    ) -> List[List[Solution]]:
        """
        Solve multiple instances.

        Args:
            instances: List of task arrays
            verbose: Print progress

        Returns:
            List of Pareto fronts, one per instance
        """
        results = []

        iterator = enumerate(instances)
        if verbose:
            iterator = tqdm(list(iterator), desc="Solving instances")

        for idx, tasks in iterator:
            pf, _ = self.solve_instance(tasks, verbose=False)
            results.append(pf)

        return results


def load_model_for_inference(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[ECSPNet, Dict]:
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model to

    Returns:
        model: Loaded ECSPNet model
        config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = ECSPNet(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_blocks=config["num_blocks"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model, config


def evaluate_on_benchmark(
    model: ECSPNet,
    N: int,
    num_instances: int = 100,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on benchmark instances.

    Args:
        model: Trained ECSPNet model
        N: Number of tasks per instance
        num_instances: Number of test instances
        device: Inference device
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    inferencer = Inferencer(model, torch.device(device))

    # Generate test instances
    instances = [generate_instance(N) for _ in range(num_instances)]

    # Solve all instances
    pareto_fronts = inferencer.solve_batch(instances, verbose=verbose)

    # Compute statistics
    all_twts = []
    all_eecs = []
    pf_sizes = []
    hypervolumes = []

    for pf in pareto_fronts:
        pf_sizes.append(len(pf))

        for sol in pf:
            all_twts.append(sol.twt)
            all_eecs.append(sol.eec)

        if len(pf) > 0:
            hv = compute_hypervolume(pf)
            hypervolumes.append(hv)

    results = {
        "N": N,
        "num_instances": num_instances,
        "mean_pf_size": np.mean(pf_sizes),
        "mean_twt": np.mean(all_twts) if all_twts else 0,
        "mean_eec": np.mean(all_eecs) if all_eecs else 0,
        "min_twt": np.min(all_twts) if all_twts else 0,
        "max_twt": np.max(all_twts) if all_twts else 0,
        "min_eec": np.min(all_eecs) if all_eecs else 0,
        "max_eec": np.max(all_eecs) if all_eecs else 0,
        "mean_hypervolume": np.mean(hypervolumes) if hypervolumes else 0,
    }

    if verbose:
        print(f"\nEvaluation results for N={N}:")
        print(
            f"  Pareto front size: {results['mean_pf_size']:.2f} ± {np.std(pf_sizes):.2f}"
        )
        print(
            f"  TWT: {results['mean_twt']:.4f} (range: {results['min_twt']:.4f} - {results['max_twt']:.4f})"
        )
        print(
            f"  EEC: {results['mean_eec']:.4f} (range: {results['min_eec']:.4f} - {results['max_eec']:.4f})"
        )
        print(f"  Hypervolume: {results['mean_hypervolume']:.4f}")

    return results


def visualize_pareto_front(
    pareto_front: List[Solution],
    all_solutions: List[Solution] = None,
    title: str = "Pareto Front",
    save_path: str = None,
):
    """
    Visualize Pareto front and optionally all solutions.

    Args:
        pareto_front: Pareto-optimal solutions
        all_solutions: All generated solutions (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all solutions (if provided)
    if all_solutions:
        twts = [s.twt for s in all_solutions]
        eecs = [s.eec for s in all_solutions]
        ax.scatter(twts, eecs, c="lightgray", alpha=0.5, label="All solutions", s=20)

    # Plot Pareto front
    pf_twts = [s.twt for s in pareto_front]
    pf_eecs = [s.eec for s in pareto_front]
    ax.scatter(pf_twts, pf_eecs, c="red", s=50, label="Pareto front", zorder=5)
    ax.plot(pf_twts, pf_eecs, "r--", alpha=0.5, linewidth=2)

    ax.set_xlabel("Total Wait Time (TWT)", fontsize=12)
    ax.set_ylabel("Energy Cost (EEC)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test inference module
    print("Testing inference module...")

    # Create a dummy model
    from .model import ECSPNet

    model = ECSPNet(d_model=128, num_heads=8, num_blocks=2)

    # Test on small instance
    N = 10
    tasks = generate_instance(N, seed=42)
    print(f"\nTest instance with {N} tasks")

    # Create inferencer with fewer solutions for testing
    inferencer = Inferencer(model, B=50, beta=0.9)

    # Solve instance
    pf, all_sols = inferencer.solve_instance(tasks, return_all=True, verbose=True)

    print(f"\nResults:")
    print(f"  Total solutions: {len(all_sols)}")
    print(f"  Pareto front size: {len(pf)}")
    print(f"\nPareto front solutions:")
    for i, sol in enumerate(pf[:10]):
        print(f"  {i+1}. TWT={sol.twt:.4f}, EEC={sol.eec:.4f}, w={sol.w:.3f}")
    if len(pf) > 10:
        print(f"  ... and {len(pf) - 10} more solutions")

    # Test Pareto front computation
    print("\nTesting Pareto dominance...")
    s1 = Solution(twt=1.0, eec=2.0, w=0.5)
    s2 = Solution(twt=1.5, eec=2.5, w=0.5)
    s3 = Solution(twt=0.8, eec=2.2, w=0.5)
    print(f"  s1 dominates s2: {s1.dominates(s2)}")  # True
    print(f"  s1 dominates s3: {s1.dominates(s3)}")  # False
    print(f"  s3 dominates s1: {s3.dominates(s1)}")  # False
