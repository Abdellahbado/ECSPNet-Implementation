"""
Data generation module for ECSP problem.
Paper-exact implementation.
"""

import numpy as np
from typing import List, Tuple

# Time discretization
DT = 0.1  # All durations are multiples of 0.1


# TOU pattern: Daily cycle of 20 slots (20 * 0.1 = 2 time units = 20 hours)
# Pattern: [High:0.6, Low:0.4, High:0.6, Low:0.4] time units
# = [6 high, 4 low, 6 high, 4 low] slots
# slots 0-5: high (1), slots 6-9: low (0), slots 10-15: high (1), slots 16-19: low (0)
def generate_tou_pattern(num_slots: int = 20) -> np.ndarray:
    """
    Generate TOU (Time-of-Use) electricity price pattern.
    Pattern repeats every 20 slots (2 time units = 20 hours).

    Returns:
        np.ndarray: Binary array where 1=high price, 0=low price
    """
    daily_cycle = 20  # slots per day
    pattern = np.zeros(num_slots, dtype=np.float32)

    for i in range(num_slots):
        slot_in_day = i % daily_cycle
        if 0 <= slot_in_day <= 5:  # High price (0.6 time units = 6 slots)
            pattern[i] = 1.0
        elif 6 <= slot_in_day <= 9:  # Low price (0.4 time units = 4 slots)
            pattern[i] = 0.0
        elif 10 <= slot_in_day <= 15:  # High price (0.6 time units = 6 slots)
            pattern[i] = 1.0
        else:  # Low price (0.4 time units = 4 slots)
            pattern[i] = 0.0

    return pattern


def get_price_at_slot(slot_idx: int) -> float:
    """Get electricity price (0 or 1) at a given slot index."""
    slot_in_day = slot_idx % 20
    if 0 <= slot_in_day <= 5:
        return 1.0
    elif 6 <= slot_in_day <= 9:
        return 0.0
    elif 10 <= slot_in_day <= 15:
        return 1.0
    else:
        return 0.0


def generate_instance(N: int, seed: int = None) -> np.ndarray:
    """
    Generate a single problem instance with N tasks.

    Each task has 5 features:
    - p1: Step 1 duration, uniformly from {0.4, 0.5, 0.6, 0.7, 0.8}
    - p2: Step 2 duration (energy-intensive), uniformly from {0.2, 0.3, 0.4, 0.5, 0.6}
    - p3: Step 3 duration, uniformly from {0.4, 0.5, 0.6, 0.7, 0.8}
    - P_high: Power during high price = 1.0 (fixed)
    - P_low: Power during low price = 0.0 (fixed, not used in paper)

    Args:
        N: Number of tasks
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape [N, 5] task features
    """
    if seed is not None:
        np.random.seed(seed)

    tasks = np.zeros((N, 5), dtype=np.float32)

    for i in range(N):
        # Generate durations with 0.1 precision
        # round(uniform(a,b), 1) gives values in {a, a+0.1, ..., b}
        p1 = np.round(np.random.uniform(0.4, 0.8), 1)
        p2 = np.round(np.random.uniform(0.2, 0.6), 1)
        p3 = np.round(np.random.uniform(0.4, 0.8), 1)

        tasks[i] = [p1, p2, p3, 1.0, 0.0]  # [p1, p2, p3, P_high, P_low]

    return tasks


def generate_batch(N: int, batch_size: int, seed: int = None) -> np.ndarray:
    """
    Generate a batch of problem instances (vectorized for speed).

    Args:
        N: Number of tasks per instance
        batch_size: Number of instances
        seed: Random seed

    Returns:
        np.ndarray: Shape [batch_size, N, 5]
    """
    if seed is not None:
        np.random.seed(seed)

    # Vectorized generation - much faster than loop
    batch = np.zeros((batch_size, N, 5), dtype=np.float32)

    # Generate all random values at once and round to 0.1
    # p1: [0.4, 0.8], p2: [0.2, 0.6], p3: [0.4, 0.8]
    batch[:, :, 0] = np.round(np.random.uniform(0.4, 0.8, (batch_size, N)), 1)  # p1
    batch[:, :, 1] = np.round(np.random.uniform(0.2, 0.6, (batch_size, N)), 1)  # p2
    batch[:, :, 2] = np.round(np.random.uniform(0.4, 0.8, (batch_size, N)), 1)  # p3
    batch[:, :, 3] = 1.0  # P_high
    batch[:, :, 4] = 0.0  # P_low

    return batch


def compute_total_processing_time(tasks: np.ndarray) -> float:
    """
    Compute T_pt = sum of all p2 (step 2 durations) for baseline normalization.

    Args:
        tasks: Shape [N, 5] or [batch, N, 5]

    Returns:
        T_pt value(s)
    """
    if tasks.ndim == 2:
        return float(np.sum(tasks[:, 1]))  # sum of p2
    else:
        return np.sum(tasks[:, :, 1], axis=1)  # [batch]


# Paper benchmark scales
BENCHMARK_SCALES = [20, 40, 60, 100]

# Training parameters from paper
TRAINING_CONFIG = {
    "epochs": 3000,
    "batch_size": 2048,
    "batches_per_epoch": 50,
    "initial_lr": 1e-3,
    "lr_decay_epochs": [1000, 2000],
    "lr_decay_factor": 0.1,
    "entropy_coef": 0.1,
    "d_model": 128,
    "num_blocks": 2,
    "num_heads": 8,
}

# Inference parameters from paper
INFERENCE_CONFIG = {
    "num_solutions": 1000,  # B
    "beta": 0.9,  # truncation parameter
}

# Environment parameters
ENV_CONFIG = {
    "dt": 0.1,
    "max_wait": 0.4,  # T_PW
    "ep_horizon": 20,  # look-ahead slots for EP
    "num_w_bins": 10,  # for baseline computation
}


if __name__ == "__main__":
    # Test data generation
    print("Testing data generation...")

    # Test TOU pattern
    tou = generate_tou_pattern(40)
    print(f"TOU pattern (40 slots): {tou}")

    # Test instance generation
    for N in BENCHMARK_SCALES:
        tasks = generate_instance(N, seed=42)
        print(f"\nN={N}: tasks shape = {tasks.shape}")
        print(f"  First task: {tasks[0]}")
        print(f"  T_pt = {compute_total_processing_time(tasks):.2f}")

    # Test batch generation
    batch = generate_batch(20, 8, seed=42)
    print(f"\nBatch shape: {batch.shape}")
