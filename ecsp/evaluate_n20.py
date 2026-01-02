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
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from scipy import stats
import warnings

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from .data import generate_instance, generate_tou_pattern, TRAINING_CONFIG, ENV_CONFIG
from .env import ECSPEnv

if TORCH_AVAILABLE:
    from .model import ECSPNet
else:  # pragma: no cover
    ECSPNet = object  # type: ignore

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


@dataclass
class TestCase:
    seed: int
    case_idx: int
    tasks: np.ndarray

    @property
    def label(self) -> str:
        return f"S{self.seed}_C{self.case_idx + 1}"


# ============================================================================
# Test Case Generation & Management
# ============================================================================


def generate_test_cases(
    save_dir: str = "test_cases",
    sampling: str = "round",
    seeds: Optional[List[int]] = None,
) -> List[TestCase]:
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

    # Backward-compatible default: old behavior seeded cases with 42,43,44.
    # We model that as base seed 42, with per-case offsets.
    if seeds is None:
        seeds = [42]

    cases: List[TestCase] = []
    for base_seed in seeds:
        for case_idx in range(NUM_TEST_CASES):
            case_seed = int(base_seed) + int(case_idx)

            # Include sampling + base_seed in filename to avoid mixing cached instances.
            case_path = os.path.join(
                save_dir, f"seed{base_seed}_case{case_idx}_n{N}_{sampling}.npy"
            )

            if os.path.exists(case_path):
                tasks = np.load(case_path)
                print(
                    f"Loaded existing test case seed={base_seed} case={case_idx} from {case_path}"
                )
            else:
                np.random.seed(case_seed)
                tasks = generate_instance(N, sampling=sampling)
                np.save(case_path, tasks)
                print(
                    f"Generated and saved test case seed={base_seed} case={case_idx} to {case_path}"
                )

            cases.append(TestCase(seed=int(base_seed), case_idx=case_idx, tasks=tasks))

    return cases


def load_test_cases(
    save_dir: str = "test_cases",
    sampling: str = "round",
    seeds: Optional[List[int]] = None,
) -> List[TestCase]:
    """Load existing test cases for a given sampling mode and seed set."""
    if seeds is None:
        seeds = [42]

    cases: List[TestCase] = []
    for base_seed in seeds:
        for case_idx in range(NUM_TEST_CASES):
            case_path = os.path.join(
                save_dir, f"seed{base_seed}_case{case_idx}_n{N}_{sampling}.npy"
            )
            if not os.path.exists(case_path):
                raise FileNotFoundError(
                    f"Test case seed={base_seed} case={case_idx} not found. Run generate_test_cases first."
                )
            cases.append(
                TestCase(
                    seed=int(base_seed),
                    case_idx=case_idx,
                    tasks=np.load(case_path),
                )
            )
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
    model: Any,
    tasks: np.ndarray,
    device: Any,
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
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for ECSPNet inference. Install torch, or run baselines-only evaluation."
        )
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

        # Track current scalarized objective value (Tchebycheff style).
        # Paper narrows objective ranges by multiplying EEC by 2 for calculation.
        current_twt, current_eec = 0.0, 0.0
        current_obj = max(w * current_twt, (1 - w) * (2.0 * current_eec))

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
                    new_obj = max(w * new_twt, (1 - w) * (2.0 * new_eec))

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
            current_obj = max(w * current_twt, (1 - w) * (2.0 * current_eec))

        twt, eec = env.get_final_metrics()
        solutions.append(Solution(twt=twt, eec=eec))

    solution_time = time.time() - start_time

    # Extract Pareto front
    all_solutions = np.array([[s.twt, s.eec] for s in solutions])
    pareto_points = get_pareto_front(all_solutions)
    pareto_solutions = [Solution(twt=p[0], eec=p[1]) for p in pareto_points]

    return ParetoFront(solutions=pareto_solutions), solution_time


# ============================================================================
# Baseline: NSGA-II (Pymoo Implementation)
# Uses Pymoo library for Python 3.12+ compatibility.
# Install: pip install pymoo
# ============================================================================

# Check for Pymoo availability
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    # Define minimal stubs so this module can be imported without pymoo.
    NSGA2 = object  # type: ignore
    ElementwiseProblem = object  # type: ignore
    Sampling = object  # type: ignore
    Crossover = object  # type: ignore
    Mutation = object  # type: ignore
    minimize = None  # type: ignore
    get_termination = None  # type: ignore
    warnings.warn(
        "Pymoo not installed. Install with: pip install pymoo\n"
        "NSGA-II baseline will use random search fallback."
    )


class ECSPProblemPymoo(ElementwiseProblem if PYMOO_AVAILABLE else object):
    """
    ECSP Problem definition for Pymoo NSGA-II.

    Decision Variables (paper-style mixed encoding):
    - A permutation of N tasks (execution order)
    - A binary vector of length N (mode per task: 0=no-wait, 1=wait)

    Objectives (both minimization):
    - f1: Total Waiting Time (TWT)
    - f2: Energy Cost (EEC)
    """

    def __init__(self, tasks: np.ndarray, n_jobs: int = N):
        """
        Initialize the ECSP problem for Pymoo.

        Args:
            tasks: Task array of shape [N, 5] with features [p1, p2, p3, P_high, P_low]
            n_jobs: Number of jobs/tasks
        """
        self.tasks = tasks.copy()
        self.n_jobs = n_jobs

        # Pymoo ElementwiseProblem setup
        # We encode as 2*N integer variables:
        # - x[:N]   : permutation values in [0, N-1] (should be a true permutation)
        # - x[N:2N] : mode bits in {0,1}
        super().__init__(
            n_var=2 * n_jobs,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.concatenate([np.zeros(n_jobs), np.zeros(n_jobs)]),
            xu=np.concatenate([np.full(n_jobs, n_jobs - 1), np.ones(n_jobs)]),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a single solution.

        Args:
            x: Decision variables [2*N] - first N are permutation ranks, next N are modes
            out: Output dictionary for objectives
        """
        n = self.n_jobs

        perm = _repair_permutation(x[:n], n)
        modes = np.clip(np.rint(x[n:]).astype(np.int64), 0, 1)

        # Evaluate using environment
        env = ECSPEnv(N=n)
        env.reset(options={"tasks": self.tasks.copy(), "w": 0.5})

        for i in range(n):
            task_idx = perm[i]
            mode = modes[task_idx]  # Mode corresponds to the task
            action = task_idx * 2 + mode
            env.step(action)

        twt, eec = env.get_final_metrics()
        out["F"] = [twt, eec]


def _repair_permutation(perm_like: np.ndarray, n: int) -> np.ndarray:
    """Repair a possibly-invalid permutation vector into a valid permutation 0..n-1."""
    v = np.rint(perm_like).astype(np.int64)
    v = np.clip(v, 0, n - 1)

    used = np.zeros(n, dtype=bool)
    repaired = np.empty(n, dtype=np.int64)

    # First pass: keep first occurrences
    missing = []
    for i in range(n):
        if not used[v[i]]:
            repaired[i] = v[i]
            used[v[i]] = True
        else:
            repaired[i] = -1

    # Collect missing values
    missing = [k for k in range(n) if not used[k]]
    mi = 0
    for i in range(n):
        if repaired[i] == -1:
            repaired[i] = missing[mi]
            mi += 1
    return repaired


def _pmx_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Partially Mapped Crossover (PMX) for permutations."""
    n = len(p1)
    a, b = np.sort(np.random.choice(n, size=2, replace=False))
    if a == b:
        return p1.copy(), p2.copy()

    def make_child(pa, pb):
        child = -np.ones(n, dtype=np.int64)
        child[a:b] = pa[a:b]

        mapping = {pb[i]: pa[i] for i in range(a, b)}

        for i in range(n):
            if a <= i < b:
                continue
            val = pb[i]
            while val in mapping and val in child[a:b]:
                val = mapping[val]
            child[i] = val

        # Fill any remaining -1 with missing values (safety)
        if np.any(child == -1):
            used = set(child[child != -1].tolist())
            missing = [k for k in range(n) if k not in used]
            mi = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = missing[mi]
                    mi += 1
        return child

    c1 = make_child(p1, p2)
    c2 = make_child(p2, p1)
    return c1, c2


class ECSPPermutationBinarySampling(Sampling):
    """Random sampling for [perm | binary] encoding."""

    def __init__(self, n_jobs: int):
        super().__init__()
        self.n_jobs = n_jobs

    def _do(self, problem, n_samples, **kwargs):
        n = self.n_jobs
        X = np.zeros((n_samples, 2 * n), dtype=np.int64)
        for i in range(n_samples):
            X[i, :n] = np.random.permutation(n)
            X[i, n:] = np.random.randint(0, 2, size=n)
        return X


class ECSPPermutationBinaryCrossover(Crossover):
    """PMX crossover for permutation part + uniform crossover for binary part."""

    def __init__(self, n_jobs: int, prob: float = 0.9):
        super().__init__(n_parents=2, n_offsprings=2)
        self.n_jobs = n_jobs
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X shape: (n_parents, n_matings, n_var)
        n = self.n_jobs
        n_matings = X.shape[1]
        off = np.empty((self.n_offsprings, n_matings, 2 * n), dtype=np.int64)

        for k in range(n_matings):
            p1 = X[0, k, :].astype(np.int64)
            p2 = X[1, k, :].astype(np.int64)

            perm1, bits1 = _repair_permutation(p1[:n], n), np.clip(p1[n:], 0, 1)
            perm2, bits2 = _repair_permutation(p2[:n], n), np.clip(p2[n:], 0, 1)

            if np.random.rand() < self.prob:
                cperm1, cperm2 = _pmx_crossover(perm1, perm2)
                mask = np.random.rand(n) < 0.5
                cbits1 = np.where(mask, bits1, bits2)
                cbits2 = np.where(mask, bits2, bits1)
            else:
                cperm1, cperm2 = perm1.copy(), perm2.copy()
                cbits1, cbits2 = bits1.copy(), bits2.copy()

            off[0, k, :n] = cperm1
            off[0, k, n:] = cbits1
            off[1, k, :n] = cperm2
            off[1, k, n:] = cbits2

        return off


class ECSPPermutationBinaryMutation(Mutation):
    """Inversion mutation for permutation + bitflip for binary."""

    def __init__(
        self,
        n_jobs: int,
        perm_mut_prob: float = None,
        bit_mut_prob: float = None,
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self.perm_mut_prob = (1.0 / n_jobs) if perm_mut_prob is None else perm_mut_prob
        self.bit_mut_prob = (1.0 / n_jobs) if bit_mut_prob is None else bit_mut_prob

    def _do(self, problem, X, **kwargs):
        n = self.n_jobs
        Y = X.astype(np.int64).copy()

        for i in range(Y.shape[0]):
            perm = _repair_permutation(Y[i, :n], n)
            bits = np.clip(Y[i, n:], 0, 1)

            # Permutation inversion mutation
            if np.random.rand() < self.perm_mut_prob:
                a, b = np.sort(np.random.choice(n, size=2, replace=False))
                if a != b:
                    perm[a:b] = perm[a:b][::-1]

            # Bitflip mutation (per-bit)
            flip = np.random.rand(n) < self.bit_mut_prob
            bits = np.where(flip, 1 - bits, bits)

            Y[i, :n] = perm
            Y[i, n:] = bits

        return Y


def run_pymoo_nsga2(
    tasks: np.ndarray,
    population_size: int,
    generations: int,
    n_jobs: int = N,
    seed: int = None,
) -> Tuple[ParetoFront, float]:
    """
    Run Pymoo NSGA-II for ECSP problem.

    Args:
        tasks: Task array [N, 5]
        population_size: NSGA-II population size
        generations: Number of generations
        n_jobs: Number of jobs
        seed: Random seed for reproducibility

    Returns:
        ParetoFront: Non-dominated solutions
        float: Runtime in seconds
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("Pymoo is not installed. Install with: pip install pymoo")

    start_time = time.time()

    # Create problem instance
    problem = ECSPProblemPymoo(tasks, n_jobs)

    # Paper-style mixed decision variables:
    # - permutation: PMX crossover + inversion mutation
    # - binary: uniform crossover + bitflip mutation
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=ECSPPermutationBinarySampling(n_jobs=n_jobs),
        crossover=ECSPPermutationBinaryCrossover(n_jobs=n_jobs, prob=0.9),
        mutation=ECSPPermutationBinaryMutation(n_jobs=n_jobs),
        eliminate_duplicates=True,
    )

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Run optimization
    termination = get_termination("n_gen", generations)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )

    solution_time = time.time() - start_time

    # Extract Pareto front from results
    if res.F is not None and len(res.F) > 0:
        pts = get_pareto_front(np.asarray(res.F, dtype=np.float64))
        pareto_solutions = [Solution(twt=float(p[0]), eec=float(p[1])) for p in pts]
    else:
        pareto_solutions = []

    return ParetoFront(solutions=pareto_solutions), solution_time


def nsga2_baseline(
    tasks: np.ndarray,
    population_size: int,
    generations: int,
) -> Tuple[ParetoFront, float]:
    """
    NSGA-II baseline using Pymoo library.

    Uses Pymoo's NSGA-II implementation (Python 3.12+ compatible).
    Falls back to a simple random search if Pymoo is not available.

    Args:
        tasks: Task array [N, 5] with features [p1, p2, p3, P_high, P_low]
        population_size: Population size for NSGA-II
        generations: Number of generations

    Returns:
        ParetoFront: Non-dominated solutions found
        float: Runtime in seconds
    """
    if PYMOO_AVAILABLE:
        return run_pymoo_nsga2(tasks, population_size, generations)
    else:
        # Fallback: simple random search (not paper-exact)
        warnings.warn(
            "Pymoo not available. Using random search fallback. "
            "Install pymoo for NSGA-II results: pip install pymoo"
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


def generate_hv_table(
    hv_results: Dict,
    methods: List[str],
    case_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate Table III style markdown table for hypervolume.
    Includes Wilcoxon test results comparing each method to ECSPNet (if present).
    """
    lines = ["# Hypervolume Results (N=20)", ""]
    lines.append("Reference point: (6, 6)")
    lines.append("")

    # Check if ECSPNet is in results for significance testing
    has_ecspnet = "ECSPNet" in hv_results and len(hv_results["ECSPNet"]) > 0

    if case_labels is None:
        n_cases = max((len(v) for v in hv_results.values()), default=0)
        case_labels = [f"Case {i + 1}" for i in range(n_cases)]
    n_cases = len(case_labels)
    header_cols = " | ".join(case_labels)

    if has_ecspnet:
        lines.append(f"| Method | {header_cols} | Mean | Sig |")
        lines.append("|--------|" + "|".join(["--------"] * n_cases) + "|------|-----|")
        ecspnet_hv = hv_results["ECSPNet"]
    else:
        lines.append(f"| Method | {header_cols} | Mean |")
        lines.append("|--------|" + "|".join(["--------"] * n_cases) + "|------|")

    for method in methods:
        if method not in hv_results:
            continue
        vals = hv_results[method]
        mean_val = np.mean(vals) if len(vals) else 0.0

        rendered = []
        for i in range(n_cases):
            rendered.append(f"{vals[i]:.4f}" if i < len(vals) else "")

        if has_ecspnet:
            # Wilcoxon test vs ECSPNet
            if method == "ECSPNet":
                sig = "-"
            else:
                _, sig = wilcoxon_test(ecspnet_hv, vals)
            line = (
                f"| {method} | " + " | ".join(rendered) + f" | {mean_val:.4f} | {sig} |"
            )
        else:
            line = f"| {method} | " + " | ".join(rendered) + f" | {mean_val:.4f} |"
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


def generate_time_table(
    time_results: Dict,
    methods: List[str],
    case_labels: Optional[List[str]] = None,
) -> str:
    """Generate solution time table."""
    lines = ["# Solution Times (N=20)", ""]
    if case_labels is None:
        n_cases = max((len(v) for v in time_results.values()), default=0)
        case_labels = [f"Case {i + 1}" for i in range(n_cases)]
    n_cases = len(case_labels)

    header_cols = " | ".join([f"{c} (s)" for c in case_labels])
    lines.append(f"| Method | {header_cols} | Mean (s) |")
    lines.append("|--------|" + "|".join(["------------"] * n_cases) + "|----------|")

    for method in methods:
        if method not in time_results:
            continue
        vals = time_results[method]
        mean_val = np.mean(vals) if len(vals) else 0.0

        rendered = []
        for i in range(n_cases):
            rendered.append(f"{vals[i]:.2f}" if i < len(vals) else "")
        line = f"| {method} | " + " | ".join(rendered) + f" | {mean_val:.2f} |"
        lines.append(line)

    return "\n".join(lines)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================


def run_full_evaluation(
    model_path: str,
    output_dir: str = "evaluation_results",
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
    baselines_only: bool = False,
    sampling: str = "round",
    seeds: Optional[List[int]] = None,
):
    """
    Run complete paper-exact evaluation for N=20.

    Args:
        model_path: Path to trained ECSPNet checkpoint
        output_dir: Directory for results
        device: Torch device
    """
    os.makedirs(output_dir, exist_ok=True)

    # If torch isn't available, we can only run baselines.
    if not TORCH_AVAILABLE and not baselines_only:
        warnings.warn("torch not available; running baselines only.")
        baselines_only = True

    if not TORCH_AVAILABLE and device != "cpu":
        warnings.warn("torch not available; forcing device='cpu'.")
        device = "cpu"

    device = torch.device(device) if TORCH_AVAILABLE else "cpu"
    if TORCH_AVAILABLE and isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; using CPU.")
            device = torch.device("cpu")

    print("=" * 60)
    print("Paper-Exact Evaluation Suite for N=20")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  B = {B_SOLUTIONS} solutions")
    print(f"  β = {BETA} (truncation)")
    print(f"  HV reference = {HV_REFERENCE}")
    print(f"  α = {SIGNIFICANCE_LEVEL} (Wilcoxon)")
    print(f"  sampling = {sampling}")
    print(f"  seeds = {seeds if seeds is not None else '[42] (default)'}")

    model = None
    if not baselines_only:
        print("\n1. Loading trained model...")
        model = ECSPNet(
            d_model=TRAINING_CONFIG["d_model"],
            num_heads=TRAINING_CONFIG["num_heads"],
            num_blocks=TRAINING_CONFIG["num_blocks"],
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"   Loaded model from {model_path}")

    # Generate/load test cases
    print("\n1. Loading test cases...")
    test_cases = generate_test_cases(
        os.path.join(output_dir, "test_cases"),
        sampling=sampling,
        seeds=seeds,
    )

    case_labels = [tc.label for tc in test_cases]

    # Results storage
    results = {
        "hypervolume": {},
        "c_metric": {},
        "solution_times": {},
        "pareto_fronts": {},
    }

    methods: List[str] = []
    if not baselines_only:
        methods.append("ECSPNet")
    methods.append("Greedy")
    for budget in MOEA_BUDGETS:
        methods.append(f"NSGA-II({budget[0]}x{budget[1]})")

    for method in methods:
        results["hypervolume"][method] = []
        results["c_metric"][method] = {"C(A,B)": [], "C(B,A)": []}
        results["solution_times"][method] = []
        results["pareto_fronts"][method] = []

    # Evaluate each test case
    for case_global_idx, tc in enumerate(test_cases):
        tasks = tc.tasks
        print(f"\n2.{case_global_idx + 1}. Evaluating {tc.label}...")

        case_fronts = {}

        if not baselines_only:
            print("   - ECSPNet inference...")
            pf_ecspnet, time_ecspnet = ecspnet_inference(model, tasks, device)
            front_ecspnet = pf_ecspnet.to_array()
            results["hypervolume"]["ECSPNet"].append(compute_hypervolume(front_ecspnet))
            results["solution_times"]["ECSPNet"].append(time_ecspnet)
            results["pareto_fronts"]["ECSPNet"].append(front_ecspnet)
            case_fronts["ECSPNet"] = front_ecspnet
            print(
                f"     HV={results['hypervolume']['ECSPNet'][-1]:.4f}, Time={time_ecspnet:.2f}s, |P|={len(front_ecspnet)}"
            )

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
            case_global_idx,
            os.path.join(output_dir, f"pareto_front_{tc.label}.png"),
        )

    # Generate tables with Wilcoxon test
    print("\n4. Generating result tables...")

    # Table III: Hypervolume with significance
    hv_table = generate_hv_table(
        results["hypervolume"], methods, case_labels=case_labels
    )
    with open(os.path.join(output_dir, "table_hypervolume.md"), "w") as f:
        f.write(hv_table)
    print(f"   Saved hypervolume table to table_hypervolume.md")

    # Table V: C-metric
    baseline_methods = [m for m in methods if m != "ECSPNet"]
    cm_table = generate_cmetric_table(results["c_metric"], baseline_methods)
    with open(os.path.join(output_dir, "table_cmetric.md"), "w") as f:
        f.write(cm_table)
    print(f"   Saved C-metric table to table_cmetric.md")

    # Solution times
    time_table = generate_time_table(
        results["solution_times"], methods, case_labels=case_labels
    )
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


def run_sampling_comparison(
    model_path: str,
    output_dir: str = "evaluation_results",
    device: str = "cpu",
    baselines_only: bool = False,
    seeds: Optional[List[int]] = None,
):
    """Run evaluation twice: sampling='round' and sampling='choice'."""
    out_round = os.path.join(output_dir, "sampling_round")
    out_choice = os.path.join(output_dir, "sampling_choice")

    print("\n=== Sampling comparison: round ===")
    res_round = run_full_evaluation(
        model_path=model_path,
        output_dir=out_round,
        device=device,
        baselines_only=baselines_only,
        sampling="round",
        seeds=seeds,
    )

    print("\n=== Sampling comparison: choice ===")
    res_choice = run_full_evaluation(
        model_path=model_path,
        output_dir=out_choice,
        device=device,
        baselines_only=baselines_only,
        sampling="choice",
        seeds=seeds,
    )

    return {"round": res_round, "choice": res_choice}


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
        "--device",
        type=str,
        default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
    )

    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Skip ECSPNet inference and run Greedy/NSGA-II only",
    )

    parser.add_argument(
        "--sampling",
        type=str,
        default="round",
        choices=["round", "choice"],
        help="Instance generation: 'round' (continuous then round to 0.1) or 'choice' (discrete-uniform on 0.1 grid)",
    )

    parser.add_argument(
        "--compare-sampling",
        action="store_true",
        help="Run both sampling modes and save under output/sampling_round and output/sampling_choice",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of base seeds to run (each base seed generates NUM_TEST_CASES cases via +case_idx)",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First base seed when using --num-seeds (defaults to 42 to match prior behavior)",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit base seeds (overrides --num-seeds/--seed-start). Example: 0,1,2,3",
    )

    args = parser.parse_args()

    if args.seeds is not None:
        base_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        base_seeds = [args.seed_start + i for i in range(args.num_seeds)]

    if args.compare_sampling:
        run_sampling_comparison(
            model_path=args.model,
            output_dir=args.output,
            device=args.device,
            baselines_only=args.baselines_only,
            seeds=base_seeds,
        )
    else:
        run_full_evaluation(
            args.model,
            args.output,
            args.device,
            baselines_only=args.baselines_only,
            sampling=args.sampling,
            seeds=base_seeds,
        )
