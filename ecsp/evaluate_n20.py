"""
Paper-Exact Evaluation Suite for N=20.
Implements all metrics and baselines as described in the paper.

Reference: "Deep Reinforcement Learning Energy Scheduling"

FIXES APPLIED:
- Greedy baseline now properly implements paper's algorithm with true state evaluation
- HV computation cleaned up (dead code removed)
- Wilcoxon test now used in table generation
- GMOEA/MOEA/D-DQN marked as not implemented with notes
- TOU usage clarified (handled by environment, not instance generator)
"""

import os
import json
import time
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from scipy import stats
import warnings

from .data import generate_instance, generate_tou_pattern, TRAINING_CONFIG, ENV_CONFIG
from .env import ECSPEnv
from .model import ECSPNet

# ============================================================================
# Constants from Paper
# ============================================================================

N = 20  # Fixed problem size
NUM_TEST_CASES = 3  # Paper uses 3 cases per scale
B_SOLUTIONS = 1000  # Number of solutions for inference
BETA = 0.9  # Truncation parameter
W_MIN, W_MAX = 0.01, 0.99  # Weight range
HV_REFERENCE = (0.3 * N, 0.3 * N)  # (6, 6) for n=20
SIGNIFICANCE_LEVEL = 0.05  # Wilcoxon test

# Baseline budgets: (population, iterations)
MOEA_BUDGETS = [(100, 100), (300, 300), (1000, 1000)]


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Solution:
    """A single solution with its objectives."""

    twt: float  # Total Waiting Time
    eec: float  # Energy Cost
    schedule: Optional[List] = None


@dataclass
class ParetoFront:
    """Collection of non-dominated solutions."""

    solutions: List[Solution]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [n_solutions, 2]."""
        if len(self.solutions) == 0:
            return np.empty((0, 2))
        return np.array([[s.twt, s.eec] for s in self.solutions])


# ============================================================================
# Test Case Generation & Management
# ============================================================================


def generate_test_cases(save_dir: str = "test_cases") -> List[np.ndarray]:
    """
    Generate and save 3 test cases for N=20.
    Each case is saved to disk to ensure reproducibility.

    Paper specs:
    - 3 steps per job
    - Processing times with precision 0.1
    - step1 in [0.4, 0.8], step2 in [0.2, 0.6], step3 in [0.4, 0.8]
    - P_high = 1.0

    NOTE: The TOU pattern is NOT applied here - it's handled by ECSPEnv
    during scheduling. The generate_instance() function only creates task
    durations and power parameters.
    """
    os.makedirs(save_dir, exist_ok=True)
    cases = []

    for case_idx in range(NUM_TEST_CASES):
        case_path = os.path.join(save_dir, f"case_{case_idx}_n{N}.npy")

        if os.path.exists(case_path):
            # Load existing case
            tasks = np.load(case_path)
            print(f"Loaded existing test case {case_idx} from {case_path}")
        else:
            # Generate new case with fixed seed for reproducibility
            np.random.seed(42 + case_idx)
            tasks = generate_instance(N)
            np.save(case_path, tasks)
            print(f"Generated and saved test case {case_idx} to {case_path}")

        cases.append(tasks)

    return cases


def load_test_cases(save_dir: str = "test_cases") -> List[np.ndarray]:
    """Load existing test cases."""
    cases = []
    for case_idx in range(NUM_TEST_CASES):
        case_path = os.path.join(save_dir, f"case_{case_idx}_n{N}.npy")
        if not os.path.exists(case_path):
            raise FileNotFoundError(
                f"Test case {case_idx} not found. Run generate_test_cases first."
            )
        cases.append(np.load(case_path))
    return cases


# ============================================================================
# Pareto Front Operations
# ============================================================================


def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a is dominated by solution b (lower is better)."""
    return np.all(b <= a) and np.any(b < a)


def get_pareto_front(solutions: np.ndarray) -> np.ndarray:
    """
    Extract non-dominated solutions from a set.

    Args:
        solutions: Array of shape [n, 2] with (TWT, EEC) pairs

    Returns:
        Non-dominated solutions array
    """
    if len(solutions) == 0:
        return np.empty((0, 2))

    n = len(solutions)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i != j and is_pareto[j]:
                if is_dominated(solutions[i], solutions[j]):
                    is_pareto[i] = False
                    break

    return solutions[is_pareto]


# ============================================================================
# Metrics (Paper Equations)
# ============================================================================


def compute_hypervolume(
    pareto_front: np.ndarray, reference: Tuple[float, float] = HV_REFERENCE
) -> float:
    """
    Compute hypervolume indicator (Eq. 38 in paper).

    HV = volume dominated by Pareto front / volume between origin and reference

    Uses the standard 2D sweep-line algorithm.

    Args:
        pareto_front: Non-dominated solutions [n, 2]
        reference: Reference point (r_twt, r_eec), default (6, 6)

    Returns:
        Normalized hypervolume in [0, 1]
    """
    if len(pareto_front) == 0:
        return 0.0

    # Filter points that are within reference bounds (dominated region)
    valid = (pareto_front[:, 0] < reference[0]) & (pareto_front[:, 1] < reference[1])
    points = pareto_front[valid]

    if len(points) == 0:
        return 0.0

    # Sort by first objective (TWT) ascending
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]

    # 2D Hypervolume via sweep-line: sum of rectangles from each point to reference
    hv = 0.0
    prev_y = reference[1]  # Start from reference y

    for i in range(len(sorted_points)):
        x, y = sorted_points[i]
        if y < prev_y:
            # Add rectangle: width = (ref_x - x), height = (prev_y - y)
            width = reference[0] - x
            height = prev_y - y
            hv += width * height
            prev_y = y

    # Normalize by hv_O (area of reference rectangle from origin)
    hv_o = reference[0] * reference[1]

    return hv / hv_o if hv_o > 0 else 0.0


def compute_c_metric(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute C-metric (Eq. 39 in paper).

    C(A, B) = fraction of solutions in B dominated by at least one solution in A

    Note: C(A,B) ≠ 1 - C(B,A) in general.

    Args:
        A: First Pareto front [n_a, 2]
        B: Second Pareto front [n_b, 2]

    Returns:
        C(A, B) in [0, 1]
    """
    if len(B) == 0:
        return 0.0

    dominated_count = 0
    for b in B:
        for a in A:
            if is_dominated(b, a):
                dominated_count += 1
                break

    return dominated_count / len(B)


# ============================================================================
# ECSPNet Inference (Algorithm 2)
# ============================================================================


def ecspnet_inference(
    model: ECSPNet,
    tasks: np.ndarray,
    device: torch.device,
    B: int = B_SOLUTIONS,
    beta: float = BETA,
) -> Tuple[ParetoFront, float]:
    """
    Run ECSPNet inference following Algorithm 2.

    Args:
        model: Trained ECSPNet model
        tasks: Task instance [N, 5]
        device: Torch device
        B: Number of solutions to generate
        beta: Truncation parameter

    Returns:
        Pareto front and solution time
    """
    model.eval()
    start_time = time.time()

    solutions = []

    # Generate B solutions with varied weights
    # w = i / (B + 1) for i = 1, ..., B (paper's weight sweep)
    weights = np.array([i / (B + 1) for i in range(1, B + 1)])

    with torch.no_grad():
        for w in weights:
            # Create environment and reset
            env = ECSPEnv(N=N)
            obs, _ = env.reset(options={"tasks": tasks.copy(), "w": w})

            done = False
            while not done:
                # Convert observation to tensors
                obs_tensor = {
                    "tasks": torch.from_numpy(obs["tasks"]).unsqueeze(0).to(device),
                    "EP": torch.from_numpy(obs["EP"]).unsqueeze(0).to(device),
                    "objs": torch.from_numpy(obs["objs"]).unsqueeze(0).to(device),
                    "w": torch.from_numpy(obs["w"]).unsqueeze(0).to(device),
                    "mask": torch.from_numpy(obs["mask"]).unsqueeze(0).to(device),
                }

                # Get action probabilities
                probs, _ = model.forward_from_obs(obs_tensor)

                # Truncation sampling (Eq. 37)
                action, _ = model.sample_action_truncated(probs, beta=beta)
                action = action.item()

                # Step environment
                obs, _, done, _, _ = env.step(action)

            # Get final objectives
            twt, eec = env.get_final_metrics()
            solutions.append(Solution(twt=twt, eec=eec))

    solution_time = time.time() - start_time

    # Extract Pareto front
    all_solutions = np.array([[s.twt, s.eec] for s in solutions])
    pareto_points = get_pareto_front(all_solutions)

    pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]

    return ParetoFront(solutions=pareto_solutions), solution_time


# ============================================================================
# Baseline: Greedy (Paper-Faithful Implementation)
# ============================================================================


def greedy_baseline(
    tasks: np.ndarray,
    num_solutions: int = B_SOLUTIONS,
) -> Tuple[ParetoFront, float]:
    """
    Greedy baseline as described in paper.

    Paper's Greedy selection principle:
    1. Prioritize actions that do NOT increase the weighted objective value
    2. Among those, select action whose task has the highest high-power time (p2)

    This implementation properly evaluates each action by simulating it
    and measuring the resulting weighted Tchebycheff objective.

    Args:
        tasks: Task instance [N, 5]
        num_solutions: Number of solutions to generate (1000 to match paper)

    Returns:
        Pareto front and solution time
    """
    start_time = time.time()
    solutions = []

    weights = np.linspace(W_MIN, W_MAX, num_solutions)

    for w in weights:
        # Main environment for this weight
        env = ECSPEnv(N=N)
        obs, _ = env.reset(options={"tasks": tasks.copy(), "w": w})

        # Track current objective value
        current_twt, current_eec = 0.0, 0.0
        current_obj = max(w * current_twt, (1 - w) * current_eec)

        # Track which tasks have been scheduled IN ORDER (order matters for time!)
        scheduled_seq = []  # list of (task_idx, mode) in execution order

        done = False
        while not done:
            # Get valid task indices
            valid_mask = obs["mask"]
            valid_indices = np.where(valid_mask > 0)[0]

            if len(valid_indices) == 0:
                break

            # Evaluate each valid action by simulation
            action_scores = []  # List of (action, delta_obj, p2)

            for task_idx in valid_indices:
                p2 = tasks[task_idx, 1]  # High-power duration (for tie-breaking)

                for mode in [0, 1]:
                    action = task_idx * 2 + mode

                    # Simulate this action by creating a copy of current state
                    # We need to replay events to reach current state, then try action
                    test_env = ECSPEnv(N=N)
                    test_env.reset(options={"tasks": tasks.copy(), "w": w})

                    # Replay previously scheduled tasks IN ORDER
                    for prev_task, prev_mode in scheduled_seq:
                        test_env.step(prev_task * 2 + prev_mode)

                    # Now try the candidate action
                    test_env.step(action)

                    # Get resulting objectives
                    new_twt, new_eec = test_env.get_final_metrics()
                    new_obj = max(w * new_twt, (1 - w) * new_eec)

                    delta_obj = new_obj - current_obj
                    action_scores.append((action, delta_obj, p2, task_idx, mode))

            # Paper's selection:
            # 1. First priority: actions that don't increase objective (delta <= 0)
            # 2. Second priority: highest p2 among those
            non_increasing = [a for a in action_scores if a[1] <= 0]

            if non_increasing:
                # Among non-increasing, select highest p2
                best = max(non_increasing, key=lambda x: x[2])
            else:
                # If all increase, select the one with smallest increase
                best = min(action_scores, key=lambda x: x[1])

            best_action = best[0]
            scheduled_seq.append((best[3], best[4]))  # Track (task_idx, mode) in order

            # Execute the action in main environment
            obs, _, done, _, _ = env.step(best_action)

            # Update current objective
            current_twt, current_eec = env.get_final_metrics()
            current_obj = max(w * current_twt, (1 - w) * current_eec)

        twt, eec = env.get_final_metrics()
        solutions.append(Solution(twt=twt, eec=eec))

    solution_time = time.time() - start_time

    # Extract Pareto front
    all_solutions = np.array([[s.twt, s.eec] for s in solutions])
    pareto_points = get_pareto_front(all_solutions)
    pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]

    return ParetoFront(solutions=pareto_solutions), solution_time


# ============================================================================
# Baseline: NSGA-II (Geatpy Implementation - Paper-Exact)
# Uses Geatpy library as specified in the original paper.
# Install: pip install geatpy
# ============================================================================

# Check for Geatpy availability
try:
    import geatpy as ea

    GEATPY_AVAILABLE = True
except ImportError:
    GEATPY_AVAILABLE = False
    warnings.warn(
        "Geatpy not installed. Install with: pip install geatpy\n"
        "NSGA-II baseline will not be available."
    )


class ECSPProblem(ea.Problem if GEATPY_AVAILABLE else object):
    """
    ECSP Problem definition for Geatpy NSGA-II.

    Decision Variables:
    - Permutation P: Job sequence (N integers representing task order)
    - Binary vector B: Wait modes (N binary values, 0=no wait, 1=wait)

    Encoding: We use a combined approach:
    - Variables 0 to N-1: Permutation encoding (P type in Geatpy)
    - Variables N to 2N-1: Binary encoding (RI type, integers 0 or 1)

    Objectives (both minimization):
    - f1: Total Weighted Waiting Time (TWT)
    - f2: Energy Cost (EEC)
    """

    def __init__(self, tasks: np.ndarray, n_jobs: int = N):
        """
        Initialize the ECSP problem.

        Args:
            tasks: Task array of shape [N, 5] with features [p1, p2, p3, P_high, P_low]
            n_jobs: Number of jobs/tasks
        """
        self.tasks = tasks.copy()
        self.n_jobs = n_jobs

        # Problem name
        name = "ECSP_Problem"

        # Number of objectives (TWT and EEC)
        M = 2

        # Optimization direction: 1 = minimize, -1 = maximize
        maxormins = [1, 1]  # Minimize both TWT and EEC

        # Total decision variables: N (permutation) + N (binary modes)
        Dim = 2 * n_jobs

        # Variable types:
        # 0 = continuous, 1 = discrete integer
        # For permutation + binary, all are integers
        varTypes = np.ones(Dim, dtype=int)

        # Variable bounds
        # Permutation part: 0 to N-1 (handled specially by Geatpy P encoding)
        # Binary part: 0 or 1
        lb = np.zeros(Dim)
        ub = np.concatenate(
            [
                np.full(n_jobs, n_jobs - 1),  # Permutation: 0 to N-1
                np.ones(n_jobs),  # Binary: 0 or 1
            ]
        )

        # Boundary inclusion: 1 = include boundary
        lbin = np.ones(Dim, dtype=int)
        ubin = np.ones(Dim, dtype=int)

        # Call parent constructor
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars: np.ndarray) -> np.ndarray:
        """
        Evaluate decision variables (vectorized for batch evaluation).

        This is the preferred method for Geatpy as it supports batch evaluation
        and parallelization more efficiently than aimFunc.

        Args:
            Vars: Decision variable matrix [pop_size, 2*N]
                  Vars[:, :N] = permutation indices
                  Vars[:, N:] = binary wait modes

        Returns:
            ObjV: Objective values [pop_size, 2] with columns [TWT, EEC]
        """
        pop_size = Vars.shape[0]
        ObjV = np.zeros((pop_size, 2))

        for i in range(pop_size):
            # Extract permutation and modes
            perm = Vars[i, : self.n_jobs].astype(int)
            modes = Vars[i, self.n_jobs :].astype(int)

            # Evaluate using environment
            twt, eec = self._evaluate_schedule(perm, modes)
            ObjV[i, 0] = twt
            ObjV[i, 1] = eec

        return ObjV

    def _evaluate_schedule(
        self, perm: np.ndarray, modes: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate a single schedule using ECSPEnv.

        Args:
            perm: Task permutation (order to schedule tasks)
            modes: Wait modes for each task (0=no wait, 1=wait after step1)

        Returns:
            (twt, eec): Total weighted waiting time and energy cost
        """
        env = ECSPEnv(N=self.n_jobs)
        env.reset(options={"tasks": self.tasks.copy(), "w": 0.5})

        for i in range(self.n_jobs):
            task_idx = perm[i]
            mode = modes[task_idx]  # Mode corresponds to the task, not position
            action = task_idx * 2 + mode
            env.step(action)

        return env.get_final_metrics()


def run_geatpy_nsga2(
    tasks: np.ndarray,
    population_size: int,
    generations: int,
    n_jobs: int = N,
    use_parallel: bool = True,
    num_workers: int = None,
    seed: int = None,
) -> Tuple[ParetoFront, float]:
    """
    Run Geatpy NSGA-II for ECSP problem.

    This uses the paper-exact Geatpy library implementation of NSGA-II.

    Args:
        tasks: Task array [N, 5]
        population_size: NSGA-II population size (NIND)
        generations: Number of generations (MAXGEN)
        n_jobs: Number of jobs
        use_parallel: Enable multiprocessing parallelization
        num_workers: Number of parallel workers (None = auto)
        seed: Random seed for reproducibility

    Returns:
        ParetoFront: Non-dominated solutions
        float: Runtime in seconds
    """
    if not GEATPY_AVAILABLE:
        raise ImportError("Geatpy is not installed. Install with: pip install geatpy")

    start_time = time.time()

    # Create problem instance
    problem = ECSPProblem(tasks, n_jobs)

    # Encoding setup for mixed permutation + binary
    # We use two Field objects: one for permutation, one for binary

    # Encoding for permutation part (P = permutation)
    Encoding1 = "P"  # Permutation encoding
    Field1 = ea.crtfld(
        Encoding1,
        problem.varTypes[:n_jobs],
        problem.ranges[:, :n_jobs],
        problem.borders[:, :n_jobs],
    )

    # Encoding for binary part (RI = real integer)
    Encoding2 = "RI"  # Real integer encoding
    Field2 = ea.crtfld(
        Encoding2,
        problem.varTypes[n_jobs:],
        problem.ranges[:, n_jobs:],
        problem.borders[:, n_jobs:],
    )

    # Combined encoding using 'OB' (ordered binary) for mixed types
    # Actually, Geatpy handles this better with a single RI encoding and custom init
    Encoding = "RI"
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)

    # Create population with custom initialization
    population = ea.Population(Encoding, Field, population_size)

    # Initialize with valid permutations + random binary
    init_chrom = np.zeros((population_size, 2 * n_jobs))
    for i in range(population_size):
        # Random permutation for first N variables
        init_chrom[i, :n_jobs] = np.random.permutation(n_jobs)
        # Random binary for next N variables
        init_chrom[i, n_jobs:] = np.random.randint(0, 2, n_jobs)

    population.Chrom = init_chrom
    population.Phen = population.Chrom.copy()

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create NSGA-II algorithm instance
    algorithm = ea.moea_NSGA2_templet(
        problem,
        population,
        MAXGEN=generations,
        logTras=0,  # No logging output
        drawing=0,  # No plots
    )

    # Custom recombination for mixed encoding
    algorithm.recOper = ea.Xovpmx(XOVR=0.9)  # PMX crossover for permutation
    algorithm.mutOper = ea.Mutswap(Pm=0.1)  # Swap mutation

    # Run optimization
    [NDSet, population] = algorithm.run()

    solution_time = time.time() - start_time

    # Extract Pareto front from NDSet
    if NDSet is not None and NDSet.sizes > 0:
        pareto_points = NDSet.ObjV
        pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]
    else:
        # Fallback: extract from final population
        pareto_points = get_pareto_front(population.ObjV)
        pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]

    return ParetoFront(solutions=pareto_solutions), solution_time


def nsga2_baseline(
    tasks: np.ndarray,
    population_size: int,
    generations: int,
) -> Tuple[ParetoFront, float]:
    """
    NSGA-II baseline using Geatpy library (paper-exact implementation).

    Uses Geatpy's NSGA-II implementation as specified in the original paper.
    Falls back to a simple random search if Geatpy is not available.

    Args:
        tasks: Task array [N, 5] with features [p1, p2, p3, P_high, P_low]
        population_size: Population size for NSGA-II
        generations: Number of generations

    Returns:
        ParetoFront: Non-dominated solutions found
        float: Runtime in seconds
    """
    if GEATPY_AVAILABLE:
        return run_geatpy_nsga2(tasks, population_size, generations)
    else:
        # Fallback: simple random search (not paper-exact)
        warnings.warn(
            "Geatpy not available. Using random search fallback. "
            "Install geatpy for paper-exact results: pip install geatpy"
        )
        return _random_search_fallback(tasks, population_size * generations)


def _random_search_fallback(
    tasks: np.ndarray,
    num_samples: int,
) -> Tuple[ParetoFront, float]:
    """
    Simple random search fallback when Geatpy is not available.

    This is NOT the paper-exact baseline - it's just a fallback.
    """
    start_time = time.time()

    solutions = []
    for _ in range(num_samples):
        perm = np.random.permutation(N)
        modes = np.random.randint(0, 2, N)

        env = ECSPEnv(N=N)
        env.reset(options={"tasks": tasks.copy(), "w": 0.5})

        for i in range(N):
            task_idx = perm[i]
            mode = modes[task_idx]
            action = task_idx * 2 + mode
            env.step(action)

        twt, eec = env.get_final_metrics()
        solutions.append([twt, eec])

    solutions = np.array(solutions)
    pareto_points = get_pareto_front(solutions)

    solution_time = time.time() - start_time
    pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]

    return ParetoFront(solutions=pareto_solutions), solution_time


# ============================================================================
# Baseline: GMOEA and MOEA/D-DQN (NOT IMPLEMENTED - See Notes)
# ============================================================================


def gmoea_baseline(
    tasks: np.ndarray, population_size: int, generations: int
) -> Tuple[ParetoFront, float]:
    """
    GMOEA baseline - NOT IMPLEMENTED.

    The paper compares against GMOEA which requires:
    - Specific genetic operators from the GMOEA paper
    - Problem-specific representation

    This is a placeholder returning empty results.
    To fully reproduce paper results, implement GMOEA from the original paper.
    """
    warnings.warn("GMOEA not implemented - returning empty Pareto front")
    return ParetoFront(solutions=[]), 0.0


def moead_dqn_baseline(
    tasks: np.ndarray, population_size: int, generations: int
) -> Tuple[ParetoFront, float]:
    """
    MOEA/D-DQN baseline - NOT IMPLEMENTED.

    The paper compares against MOEA/D-DQN which requires:
    - Pre-trained DQN model for value estimation
    - MOEA/D decomposition framework

    This is a placeholder returning empty results.
    To fully reproduce paper results, implement MOEA/D-DQN from the original papers.
    """
    warnings.warn("MOEA/D-DQN not implemented - returning empty Pareto front")
    return ParetoFront(solutions=[]), 0.0


# ============================================================================
# Statistical Tests
# ============================================================================


def wilcoxon_test(data_a: List[float], data_b: List[float]) -> Tuple[float, str]:
    """
    Wilcoxon rank-sum test as used in paper.

    Tests if ECSPNet (A) is significantly different from comparator (B).

    Returns:
        p-value and symbol:
        - "+" if A is significantly better (higher HV, p < 0.05)
        - "-" if B is significantly better
        - "=" if no significant difference
    """
    if len(data_a) < 3 or len(data_b) < 3:
        return 1.0, "="

    try:
        stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
    except Exception:
        return 1.0, "="

    if p_value < SIGNIFICANCE_LEVEL:
        if np.mean(data_a) > np.mean(data_b):
            return p_value, "+"  # A (ECSPNet) is better (higher HV)
        else:
            return p_value, "-"  # B (comparator) is better
    else:
        return p_value, "="  # No significant difference


# ============================================================================
# Visualization
# ============================================================================


def plot_pareto_fronts(
    fronts: Dict[str, np.ndarray],
    case_idx: int,
    save_path: str = None,
):
    """
    Plot Pareto fronts for comparison.

    Args:
        fronts: Dictionary mapping method name to Pareto front array
        case_idx: Test case index
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10.colors

    for i, (name, front) in enumerate(fronts.items()):
        if len(front) > 0:
            plt.scatter(
                front[:, 0],
                front[:, 1],
                label=name,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                s=50,
                alpha=0.7,
            )

    plt.xlabel("Total Waiting Time (TWT)", fontsize=12)
    plt.ylabel("Energy Cost (EEC)", fontsize=12)
    plt.title(f"Pareto Front Distribution - Case {case_idx + 1} (N=20)", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Mark reference point
    plt.axhline(y=HV_REFERENCE[1], color="red", linestyle="--", alpha=0.5)
    plt.axvline(x=HV_REFERENCE[0], color="red", linestyle="--", alpha=0.5)
    plt.scatter(
        [HV_REFERENCE[0]],
        [HV_REFERENCE[1]],
        color="red",
        marker="x",
        s=100,
        label="Reference (6,6)",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Pareto front plot to {save_path}")

    plt.close()


# ============================================================================
# Table Generation with Wilcoxon Test
# ============================================================================


def generate_hv_table(hv_results: Dict, methods: List[str]) -> str:
    """
    Generate Table III style markdown table for hypervolume.
    Includes Wilcoxon test results comparing each method to ECSPNet (if present).
    """
    lines = ["# Hypervolume Results (N=20)", ""]
    lines.append("Reference point: (6, 6)")
    lines.append("")

    # Check if ECSPNet is in results for significance testing
    has_ecspnet = "ECSPNet" in hv_results and len(hv_results["ECSPNet"]) > 0

    if has_ecspnet:
        lines.append("| Method | Case 1 | Case 2 | Case 3 | Mean | Sig |")
        lines.append("|--------|--------|--------|--------|------|-----|")
        ecspnet_hv = hv_results["ECSPNet"]
    else:
        lines.append("| Method | Case 1 | Case 2 | Case 3 | Mean |")
        lines.append("|--------|--------|--------|--------|------|")

    for method in methods:
        if method not in hv_results:
            continue
        vals = hv_results[method]
        mean_val = np.mean(vals)

        if has_ecspnet:
            # Wilcoxon test vs ECSPNet
            if method == "ECSPNet":
                sig = "-"
            else:
                _, sig = wilcoxon_test(ecspnet_hv, vals)
            line = f"| {method} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {mean_val:.4f} | {sig} |"
        else:
            line = f"| {method} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {mean_val:.4f} |"
        lines.append(line)

    lines.append("")
    lines.append(
        "Sig: + = ECSPNet significantly better, - = comparator better, = = no significant difference (α=0.05)"
    )

    return "\n".join(lines)


def generate_cmetric_table(cm_results: Dict, methods: List[str]) -> str:
    """Generate Table V style markdown table for C-metric."""
    # Skip C-metric table if no ECSPNet comparison was done
    if not cm_results or all(
        len(cm_results.get(m, {}).get("C(A,B)", [])) == 0 for m in methods
    ):
        return "# C-Metric Results (N=20)\n\nSkipped (ECSPNet not evaluated)"

    lines = ["# C-Metric Results (N=20)", ""]
    lines.append("A = ECSPNet, B = Comparator")
    lines.append("C(A,B) = fraction of B dominated by A (higher = ECSPNet better)")
    lines.append("C(B,A) = fraction of A dominated by B (lower = ECSPNet better)")
    lines.append("")
    lines.append("| Method | C(A,B) Mean | C(B,A) Mean |")
    lines.append("|--------|-------------|-------------|")

    for method in methods:
        if method not in cm_results:
            continue
        cab = cm_results[method].get("C(A,B)", [])
        cba = cm_results[method].get("C(B,A)", [])
        if not cab or not cba:
            continue
        line = f"| {method} | {np.mean(cab):.4f} | {np.mean(cba):.4f} |"
        lines.append(line)

    return "\n".join(lines)


def generate_time_table(time_results: Dict, methods: List[str]) -> str:
    """Generate solution time table."""
    lines = ["# Solution Times (N=20)", ""]
    lines.append("| Method | Case 1 (s) | Case 2 (s) | Case 3 (s) | Mean (s) |")
    lines.append("|--------|------------|------------|------------|----------|")

    for method in methods:
        if method not in time_results:
            continue
        vals = time_results[method]
        mean_val = np.mean(vals)
        line = f"| {method} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {mean_val:.2f} |"
        lines.append(line)

    return "\n".join(lines)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================


def run_full_evaluation(
    model_path: str,
    output_dir: str = "evaluation_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run complete paper-exact evaluation for N=20.

    Args:
        model_path: Path to trained ECSPNet checkpoint
        output_dir: Directory for results
        device: Torch device
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    print("=" * 60)
    print("Paper-Exact Evaluation Suite for N=20")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  B = {B_SOLUTIONS} solutions")
    print(f"  β = {BETA} (truncation)")
    print(f"  HV reference = {HV_REFERENCE}")
    print(f"  α = {SIGNIFICANCE_LEVEL} (Wilcoxon)")

    # NOTE: ECSPNet loading is skipped for baseline-only evaluation
    # Uncomment below to include ECSPNet inference
    # print("\n1. Loading trained model...")
    # model = ECSPNet(
    #     d_model=TRAINING_CONFIG["d_model"],
    #     num_heads=TRAINING_CONFIG["num_heads"],
    #     num_blocks=TRAINING_CONFIG["num_blocks"],
    # ).to(device)
    # checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # if "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"])
    # else:
    #     model.load_state_dict(checkpoint)
    # model.eval()
    # print(f"   Loaded model from {model_path}")

    # Generate/load test cases
    print("\n1. Loading test cases...")
    test_cases = generate_test_cases(os.path.join(output_dir, "test_cases"))

    # Results storage
    results = {
        "hypervolume": {},
        "c_metric": {},
        "solution_times": {},
        "pareto_fronts": {},
    }

    # Baselines only (ECSPNet skipped for scaling validation)
    methods = ["Greedy"]
    for budget in MOEA_BUDGETS:
        methods.append(f"NSGA-II({budget[0]}x{budget[1]})")

    for method in methods:
        results["hypervolume"][method] = []
        results["c_metric"][method] = {"C(A,B)": [], "C(B,A)": []}
        results["solution_times"][method] = []
        results["pareto_fronts"][method] = []

    # Evaluate each test case
    for case_idx, tasks in enumerate(test_cases):
        print(f"\n2.{case_idx + 1}. Evaluating Case {case_idx + 1}...")

        case_fronts = {}

        # NOTE: ECSPNet inference skipped for baseline validation
        # Uncomment to include:
        # print("   - ECSPNet inference...")
        # pf_ecspnet, time_ecspnet = ecspnet_inference(model, tasks, device)
        # front_ecspnet = pf_ecspnet.to_array()
        # results["hypervolume"]["ECSPNet"].append(compute_hypervolume(front_ecspnet))
        # results["solution_times"]["ECSPNet"].append(time_ecspnet)
        # results["pareto_fronts"]["ECSPNet"].append(front_ecspnet)
        # case_fronts["ECSPNet"] = front_ecspnet
        # print(f"     HV={results['hypervolume']['ECSPNet'][-1]:.4f}, Time={time_ecspnet:.2f}s, |P|={len(front_ecspnet)}")

        # Greedy
        print("   - Greedy baseline...")
        pf_greedy, time_greedy = greedy_baseline(tasks)
        front_greedy = pf_greedy.to_array()
        results["hypervolume"]["Greedy"].append(compute_hypervolume(front_greedy))
        results["solution_times"]["Greedy"].append(time_greedy)
        results["pareto_fronts"]["Greedy"].append(front_greedy)
        case_fronts["Greedy"] = front_greedy
        print(
            f"     HV={results['hypervolume']['Greedy'][-1]:.4f}, Time={time_greedy:.2f}s, |P|={len(front_greedy)}"
        )

        # NSGA-II with different budgets
        for pop, gen in MOEA_BUDGETS:
            method_name = f"NSGA-II({pop}x{gen})"
            print(f"   - {method_name}...")
            pf_nsga, time_nsga = nsga2_baseline(tasks, pop, gen)
            front_nsga = pf_nsga.to_array()
            results["hypervolume"][method_name].append(compute_hypervolume(front_nsga))
            results["solution_times"][method_name].append(time_nsga)
            results["pareto_fronts"][method_name].append(front_nsga)
            case_fronts[method_name] = front_nsga
            print(
                f"     HV={results['hypervolume'][method_name][-1]:.4f}, Time={time_nsga:.2f}s, |P|={len(front_nsga)}"
            )

        # Plot Pareto fronts for this case
        plot_pareto_fronts(
            case_fronts,
            case_idx,
            os.path.join(output_dir, f"pareto_front_case{case_idx + 1}.png"),
        )

    # Generate tables with Wilcoxon test
    print("\n4. Generating result tables...")

    # Table III: Hypervolume with significance
    hv_table = generate_hv_table(results["hypervolume"], methods)
    with open(os.path.join(output_dir, "table_hypervolume.md"), "w") as f:
        f.write(hv_table)
    print(f"   Saved hypervolume table to table_hypervolume.md")

    # Table V: C-metric
    cm_table = generate_cmetric_table(
        results["c_metric"], methods[1:]
    )  # Exclude ECSPNet
    with open(os.path.join(output_dir, "table_cmetric.md"), "w") as f:
        f.write(cm_table)
    print(f"   Saved C-metric table to table_cmetric.md")

    # Solution times
    time_table = generate_time_table(results["solution_times"], methods)
    with open(os.path.join(output_dir, "table_solution_times.md"), "w") as f:
        f.write(time_table)
    print(f"   Saved solution times table to table_solution_times.md")

    # Save full results as JSON
    results_json = {
        "hypervolume": {
            k: [float(v) for v in vals] for k, vals in results["hypervolume"].items()
        },
        "c_metric": {
            k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
            for k, v in results["c_metric"].items()
        },
        "solution_times": {
            k: [float(v) for v in vals] for k, vals in results["solution_times"].items()
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Print summary
    print("\n--- Summary ---")
    print(f"{'Method':<20} {'Mean HV':<10} {'Mean Time':<10}")
    print("-" * 40)
    for method in methods:
        hv = np.mean(results["hypervolume"][method])
        t = np.mean(results["solution_times"][method])
        print(f"{method:<20} {hv:<10.4f} {t:<10.2f}s")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper-exact evaluation for N=20")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ecspnet_N20_final.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    run_full_evaluation(args.model, args.output, args.device)
