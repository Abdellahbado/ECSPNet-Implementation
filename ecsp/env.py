"""
ECSP Environment - Gym-style implementation.
Paper-exact implementation of the Energy-Conscious Scheduling Problem.

VERSION: 2.0-GPU - Full GPU acceleration for training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from .data import (
    generate_instance,
    generate_tou_pattern,
    get_price_at_slot,
    DT,
    ENV_CONFIG,
)

# Version identifier for tracking
ENV_VERSION = "2.0-GPU"
print(f"[ECSPEnv v{ENV_VERSION}] Loading GPU-accelerated environment...")


@dataclass
class TaskScheduleResult:
    """Result of scheduling a single task."""

    task_idx: int
    mode: int  # 0=no wait, 1=wait after step1
    start_time: float
    step1_end: float
    wait_time: float
    step2_start: float
    step2_end: float
    step3_end: float
    eec_contribution: float  # Energy cost for this task's step2


class ECSPEnv(gym.Env):
    """
    Energy-Conscious Scheduling Problem Environment.

    State space:
        - tasks: [N, 5] remaining task features [p1, p2, p3, P_high=1, P_low=0]
        - EP: [20] next 20 time slots electricity price {0, 1}
        - objs: [2] current [TWT, EEC] (raw values)
        - w: [1] preference weight
        - mask: [N] 1=available task, 0=scheduled/padding

    Action space:
        Categorical over 2*N actions: (task_idx, mode)
        - action_idx // 2 = task_idx
        - action_idx % 2 = mode (0=no wait, 1=wait after step1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        N: int = 20,
        max_N: int = None,
        dt: float = DT,
        max_wait: float = ENV_CONFIG["max_wait"],
        ep_horizon: int = ENV_CONFIG["ep_horizon"],
    ):
        """
        Initialize ECSP environment.

        Args:
            N: Number of tasks in each instance
            max_N: Maximum N for padding (default: N)
            dt: Time discretization step
            max_wait: Maximum wait time T_PW (paper: 0.4)
            ep_horizon: Number of look-ahead slots for EP (paper: 20)
        """
        super().__init__()

        self.N = N
        self.max_N = max_N if max_N is not None else N
        self.dt = dt
        self.max_wait = max_wait
        self.ep_horizon = ep_horizon

        # Action space: 2*max_N discrete actions
        self.action_space = spaces.Discrete(2 * self.max_N)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Box(
                    low=0, high=2, shape=(self.max_N, 5), dtype=np.float32
                ),
                "EP": spaces.Box(
                    low=0, high=1, shape=(self.ep_horizon,), dtype=np.float32
                ),
                "objs": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32),
                "w": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "mask": spaces.Box(
                    low=0, high=1, shape=(self.max_N,), dtype=np.float32
                ),
            }
        )

        # State variables (initialized in reset)
        self.tasks: Optional[np.ndarray] = None
        self.original_tasks: Optional[np.ndarray] = None
        self.current_time: float = 0.0
        self.twt: float = 0.0  # Total Wait Time
        self.eec: float = 0.0  # Energy cost (sum during step2)
        self.w: float = 0.5  # Preference weight
        self.available_mask: Optional[np.ndarray] = None
        self.schedule_history: List[TaskScheduleResult] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment with a new instance.

        Options:
            - tasks: np.ndarray [N, 5] - use provided tasks instead of generating
            - w: float - preference weight (default: random uniform(0.01, 0.99))
        """
        super().reset(seed=seed)

        # Generate or use provided tasks
        if options is not None and "tasks" in options:
            self.original_tasks = options["tasks"].copy()
        else:
            self.original_tasks = generate_instance(self.N, seed=seed)

        # Pad tasks to max_N if necessary
        actual_N = len(self.original_tasks)
        self.tasks = np.zeros((self.max_N, 5), dtype=np.float32)
        self.tasks[:actual_N] = self.original_tasks

        # Initialize mask (1 for real tasks, 0 for padding)
        self.available_mask = np.zeros(self.max_N, dtype=np.float32)
        self.available_mask[:actual_N] = 1.0

        # Set preference weight
        if options is not None and "w" in options:
            self.w = options["w"]
        else:
            self.w = np.random.uniform(0.01, 0.99)

        # Reset state
        self.current_time = 0.0
        self.twt = 0.0
        self.eec = 0.0
        self.schedule_history = []

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary."""
        # Get electricity prices for next ep_horizon slots
        current_slot = int(self.current_time / self.dt)
        EP = np.array(
            [get_price_at_slot(current_slot + i) for i in range(self.ep_horizon)],
            dtype=np.float32,
        )

        return {
            "tasks": self.tasks.copy(),
            "EP": EP,
            "objs": np.array([self.twt, self.eec], dtype=np.float32),
            "w": np.array([self.w], dtype=np.float32),
            "mask": self.available_mask.copy(),
        }

    def _get_info(self) -> Dict:
        """Get additional info."""
        return {
            "current_time": self.current_time,
            "num_scheduled": len(self.schedule_history),
            "num_remaining": int(np.sum(self.available_mask)),
        }

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one scheduling action.

        Args:
            action: Integer in [0, 2*max_N), encoding (task_idx, mode)
                   task_idx = action // 2
                   mode = action % 2 (0=no wait, 1=wait after step1)

        Returns:
            observation, reward, terminated, truncated, info
        """
        task_idx = action // 2
        mode = action % 2

        # Validate action
        if self.available_mask[task_idx] == 0:
            raise ValueError(
                f"Task {task_idx} is not available (already scheduled or padding)"
            )

        # Get task features
        p1, p2, p3, P_high, P_low = self.tasks[task_idx]

        # Schedule the task
        result = self._simulate_task(task_idx, p1, p2, p3, mode)
        self.schedule_history.append(result)

        # Update state
        self.current_time = result.step3_end
        self.twt += result.wait_time
        self.eec += result.eec_contribution

        # Mark task as scheduled
        self.available_mask[task_idx] = 0.0

        # Check if episode is done
        terminated = np.sum(self.available_mask) == 0
        truncated = False

        # Compute reward only at episode end
        if terminated:
            # Scale EEC by 2 as per paper
            final_eec = self.eec * 2
            # Reward: R = -max(w*TWT, (1-w)*EEC)
            reward = -max(self.w * self.twt, (1 - self.w) * final_eec)
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _simulate_task(
        self, task_idx: int, p1: float, p2: float, p3: float, mode: int
    ) -> TaskScheduleResult:
        """
        Simulate scheduling a task with given mode.

        Args:
            task_idx: Index of the task
            p1, p2, p3: Step durations
            mode: 0=no wait, 1=wait after step1 (capped at max_wait)

        Returns:
            TaskScheduleResult with timing and energy cost
        """
        start_time = self.current_time

        # Step 1: always runs immediately
        step1_end = start_time + p1

        # Determine wait time
        wait_time = 0.0
        if mode == 1:
            # Check if we're in high-price period after step1
            slot_after_step1 = int(step1_end / self.dt)
            price_after_step1 = get_price_at_slot(slot_after_step1)

            if price_after_step1 == 1.0:  # High price, try to wait
                # Find distance to next low-price slot
                time_to_next_low = self._time_to_next_low_price(step1_end)
                # Cap wait at max_wait
                wait_time = min(time_to_next_low, self.max_wait)
            # If already low price, no wait needed

        # Step 2: starts after wait
        step2_start = step1_end + wait_time
        step2_end = step2_start + p2

        # Compute energy cost for step2 (sum of prices across slots)
        eec_contribution = self._compute_step2_energy_cost(step2_start, step2_end)

        # Step 3: immediately after step2
        step3_end = step2_end + p3

        return TaskScheduleResult(
            task_idx=task_idx,
            mode=mode,
            start_time=start_time,
            step1_end=step1_end,
            wait_time=wait_time,
            step2_start=step2_start,
            step2_end=step2_end,
            step3_end=step3_end,
            eec_contribution=eec_contribution,
        )

    def _time_to_next_low_price(self, current_time: float) -> float:
        """
        Compute time until next low-price slot begins.

        Args:
            current_time: Current time in continuous units

        Returns:
            Time until next slot with price=0
        """
        current_slot = int(current_time / self.dt)

        # Search ahead for low-price slot (max 20 slots = 1 full cycle)
        for offset in range(1, 21):
            if get_price_at_slot(current_slot + offset) == 0.0:
                # Found low-price slot
                next_low_slot_start = (current_slot + offset) * self.dt
                return next_low_slot_start - current_time

        # Should never reach here given the 20-slot cycle
        return self.max_wait

    def _compute_step2_energy_cost(self, step2_start: float, step2_end: float) -> float:
        """
        Compute energy cost during step2 execution.

        EEC = sum over slots k that overlap with step2 of: PH[k] * P_high * dt
        where P_high = 1.0

        Args:
            step2_start: Start time of step2
            step2_end: End time of step2

        Returns:
            Energy cost contribution from this step2
        """
        # Find slots that overlap with step2
        start_slot = int(step2_start / self.dt)
        end_slot = int(np.ceil(step2_end / self.dt))

        eec = 0.0
        for slot in range(start_slot, end_slot):
            slot_start = slot * self.dt
            slot_end = (slot + 1) * self.dt

            # Compute overlap with step2
            overlap_start = max(step2_start, slot_start)
            overlap_end = min(step2_end, slot_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                price = get_price_at_slot(slot)
                # EEC contribution: price * P_high * overlap_duration
                # P_high = 1.0, so just price * overlap
                eec += price * 1.0 * overlap_duration

        return eec

    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions.

        Returns:
            np.ndarray of shape [2*max_N] with 1 for valid actions, 0 otherwise
        """
        valid = np.zeros(2 * self.max_N, dtype=np.float32)
        for i in range(self.max_N):
            if self.available_mask[i] == 1.0:
                valid[2 * i] = 1.0  # mode=0
                valid[2 * i + 1] = 1.0  # mode=1
        return valid

    def get_final_metrics(self) -> Tuple[float, float]:
        """
        Get final TWT and raw EEC (unscaled for evaluation).

        Returns:
            (TWT, EEC) - raw values matching paper's HV reference (0.3N, 0.3N)
        Note: The ×2 scaling is kept in step() reward only, not here.
        """
        return self.twt, self.eec


class BatchECSPEnv:
    """
    Batched version of ECSPEnv for efficient training.
    Handles multiple instances in parallel.
    """

    def __init__(
        self,
        N: int = 20,
        batch_size: int = 2048,
        max_N: int = None,
        dt: float = DT,
        max_wait: float = ENV_CONFIG["max_wait"],
        ep_horizon: int = ENV_CONFIG["ep_horizon"],
    ):
        self.N = N
        self.batch_size = batch_size
        self.max_N = max_N if max_N is not None else N
        self.dt = dt
        self.max_wait = max_wait
        self.ep_horizon = ep_horizon

        # Precompute one TOU cycle (paper: 0.6/0.4/0.6/0.4 with dt=0.1 => 20 slots)
        self._cycle_len = 20
        self._price_cycle = np.array(
            [get_price_at_slot(i) for i in range(self._cycle_len)], dtype=np.float32
        )

        # For each slot-in-cycle s, store the smallest offset>=1 to reach a low-price slot.
        # Used to implement "wait until next low-price slot" efficiently.
        next_low_offset = np.zeros(self._cycle_len, dtype=np.int32)
        for s in range(self._cycle_len):
            for offset in range(1, self._cycle_len + 1):
                if self._price_cycle[(s + offset) % self._cycle_len] == 0.0:
                    next_low_offset[s] = offset
                    break
        self._next_low_offset = next_low_offset

        # Step-2 durations are multiples of dt in [0.2, 0.6] => 2..6 slots.
        self._max_p2_slots = int(round(0.6 / self.dt))
        eec_table = np.zeros(
            (self._cycle_len, self._max_p2_slots + 1), dtype=np.float32
        )
        for start_mod in range(self._cycle_len):
            for L in range(self._max_p2_slots + 1):
                if L == 0:
                    eec_table[start_mod, L] = 0.0
                else:
                    # Exact because all times are multiples of dt.
                    idxs = (start_mod + np.arange(L, dtype=np.int32)) % self._cycle_len
                    eec_table[start_mod, L] = (
                        float(self._price_cycle[idxs].sum()) * self.dt
                    )
        self._eec_table = eec_table

        # Batch state
        self.tasks: Optional[np.ndarray] = None  # [batch, max_N, 5]
        self.current_times: Optional[np.ndarray] = None  # [batch]
        self.twts: Optional[np.ndarray] = None  # [batch]
        self.eecs: Optional[np.ndarray] = None  # [batch]
        self.ws: Optional[np.ndarray] = None  # [batch]
        self.masks: Optional[np.ndarray] = None  # [batch, max_N]
        self.dones: Optional[np.ndarray] = None  # [batch]
        self.remaining: Optional[np.ndarray] = (
            None  # [batch] remaining unscheduled tasks
        )

    def reset(
        self,
        tasks: Optional[np.ndarray] = None,
        ws: Optional[np.ndarray] = None,
        seed: int = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reset all environments in batch.

        Args:
            tasks: [batch, N, 5] task features (optional)
            ws: [batch] preference weights (optional)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate or use provided tasks
        if tasks is not None:
            self.tasks = np.zeros((self.batch_size, self.max_N, 5), dtype=np.float32)
            actual_N = tasks.shape[1]
            self.tasks[:, :actual_N] = tasks
        else:
            from .data import generate_batch

            self.tasks = np.zeros((self.batch_size, self.max_N, 5), dtype=np.float32)
            generated = generate_batch(self.N, self.batch_size)
            self.tasks[:, : self.N] = generated

        # Initialize masks
        actual_N = self.N if tasks is None else tasks.shape[1]
        self.masks = np.zeros((self.batch_size, self.max_N), dtype=np.float32)
        self.masks[:, :actual_N] = 1.0
        self.remaining = np.full(self.batch_size, actual_N, dtype=np.int32)

        # Set preference weights
        if ws is not None:
            self.ws = ws.copy()
        else:
            self.ws = np.random.uniform(0.01, 0.99, size=self.batch_size).astype(
                np.float32
            )

        # Reset state
        self.current_times = np.zeros(self.batch_size, dtype=np.float32)
        self.twts = np.zeros(self.batch_size, dtype=np.float32)
        self.eecs = np.zeros(self.batch_size, dtype=np.float32)
        self.dones = np.zeros(self.batch_size, dtype=bool)

        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get batched observations."""
        # Compute EP for each instance based on current time (vectorized)
        current_slots = (self.current_times / self.dt).astype(np.int64)  # [batch]
        offsets = np.arange(self.ep_horizon, dtype=np.int64)[None, :]  # [1, H]
        slot_idxs = (current_slots[:, None] + offsets) % self._cycle_len  # [batch, H]
        EP = self._price_cycle[slot_idxs].astype(np.float32)

        return {
            "tasks": self.tasks.copy(),
            "EP": EP,
            "objs": np.stack([self.twts, self.eecs], axis=1),
            "w": self.ws[:, np.newaxis],
            "mask": self.masks.copy(),
        }

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Execute batch of actions.

        Args:
            actions: [batch] integer actions

        Returns:
            observations, rewards, dones, info
        """
        rewards = np.zeros(self.batch_size, dtype=np.float32)

        if actions.shape[0] != self.batch_size:
            raise ValueError(
                f"Expected actions shape [{self.batch_size}], got {actions.shape}"
            )

        active = ~self.dones
        if not np.any(active):
            return self._get_obs(), rewards, self.dones.copy(), {}

        active_idx = np.nonzero(active)[0]
        act = actions[active_idx].astype(np.int64)
        task_idx = (act // 2).astype(np.int64)
        mode = (act % 2).astype(np.int64)

        # Validate actions
        if np.any(self.masks[active_idx, task_idx] == 0.0):
            bad_local = np.nonzero(self.masks[active_idx, task_idx] == 0.0)[0][0]
            b = int(active_idx[bad_local])
            t = int(task_idx[bad_local])
            raise ValueError(f"Batch {b}: Task {t} not available")

        # Gather task durations
        p1 = self.tasks[active_idx, task_idx, 0]
        p2 = self.tasks[active_idx, task_idx, 1]
        p3 = self.tasks[active_idx, task_idx, 2]

        start_time = self.current_times[active_idx]
        step1_end = start_time + p1

        # Wait logic: if mode==1 and price right after step1 is high, wait until next low slot, capped.
        slot_after_step1 = (step1_end / self.dt).astype(np.int64)
        slot_mod = (slot_after_step1 % self._cycle_len).astype(np.int64)
        price_after = self._price_cycle[slot_mod]

        wait_time = np.zeros_like(step1_end, dtype=np.float32)
        need_wait = (mode == 1) & (price_after == 1.0)
        if np.any(need_wait):
            offsets = self._next_low_offset[slot_mod[need_wait]].astype(np.float32)
            time_to_low = offsets * self.dt
            wait_time[need_wait] = np.minimum(time_to_low, self.max_wait)

        step2_start = step1_end + wait_time

        # Step2 energy cost is exact because all times are multiples of dt.
        start_slot2 = (step2_start / self.dt).astype(np.int64)
        start_mod2 = (start_slot2 % self._cycle_len).astype(np.int64)
        p2_slots = np.rint(p2 / self.dt).astype(np.int64)
        p2_slots = np.clip(p2_slots, 0, self._max_p2_slots)
        eec = self._eec_table[start_mod2, p2_slots]

        step3_end = step2_start + p2 + p3

        # Update state
        self.current_times[active_idx] = step3_end
        self.twts[active_idx] += wait_time
        self.eecs[active_idx] += eec
        self.masks[active_idx, task_idx] = 0.0
        self.remaining[active_idx] -= 1

        done_now = self.remaining[active_idx] <= 0
        if np.any(done_now):
            done_global = active_idx[done_now]
            self.dones[done_global] = True
            final_eec = self.eecs[done_global] * 2
            rewards[done_global] = -np.maximum(
                self.ws[done_global] * self.twts[done_global],
                (1.0 - self.ws[done_global]) * final_eec,
            ).astype(np.float32)

        return self._get_obs(), rewards, self.dones.copy(), {}

    def _time_to_next_low(self, current_time: float) -> float:
        """Time until next low-price slot."""
        current_slot = int(current_time / self.dt)
        for offset in range(1, 21):
            if get_price_at_slot(current_slot + offset) == 0.0:
                return (current_slot + offset) * self.dt - current_time
        return self.max_wait

    def _compute_step2_eec(self, step2_start: float, step2_end: float) -> float:
        """Compute energy cost during step2."""
        start_slot = int(step2_start / self.dt)
        end_slot = int(np.ceil(step2_end / self.dt))

        eec = 0.0
        for slot in range(start_slot, end_slot):
            slot_start = slot * self.dt
            slot_end = (slot + 1) * self.dt
            overlap_start = max(step2_start, slot_start)
            overlap_end = min(step2_end, slot_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                eec += get_price_at_slot(slot) * overlap
        return eec

    def get_valid_actions_mask(self) -> np.ndarray:
        """Get valid action masks for all instances. Shape: [batch, 2*max_N]"""
        valid = np.zeros((self.batch_size, 2 * self.max_N), dtype=np.float32)
        for b in range(self.batch_size):
            for i in range(self.max_N):
                if self.masks[b, i] == 1.0:
                    valid[b, 2 * i] = 1.0
                    valid[b, 2 * i + 1] = 1.0
        return valid

    def get_final_metrics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get final (TWT, raw EEC) for all instances."""
        return self.twts.copy(), self.eecs.copy()


class GPUBatchECSPEnv:
    """
    GPU-native batched ECSP environment using PyTorch tensors.
    All operations stay on GPU to eliminate CPU-GPU transfer bottleneck.

    VERSION: 2.0-GPU
    """

    def __init__(
        self,
        N: int = 20,
        batch_size: int = 2048,
        device: torch.device = None,
        dt: float = DT,
        max_wait: float = ENV_CONFIG["max_wait"],
        ep_horizon: int = ENV_CONFIG["ep_horizon"],
    ):
        self.N = N
        self.batch_size = batch_size
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dt = dt
        self.max_wait = max_wait
        self.ep_horizon = ep_horizon
        self._cycle_len = 20

        print(
            f"[GPUBatchECSPEnv v{ENV_VERSION}] Initialized on {self.device} with batch_size={batch_size}, N={N}"
        )

        # Precompute TOU cycle on GPU
        price_cycle_np = np.array(
            [get_price_at_slot(i) for i in range(self._cycle_len)], dtype=np.float32
        )
        self._price_cycle = torch.from_numpy(price_cycle_np).to(self.device)

        # Precompute next-low-slot offsets on GPU
        next_low_offset = np.zeros(self._cycle_len, dtype=np.int64)
        for s in range(self._cycle_len):
            for offset in range(1, self._cycle_len + 1):
                if price_cycle_np[(s + offset) % self._cycle_len] == 0.0:
                    next_low_offset[s] = offset
                    break
        self._next_low_offset = torch.from_numpy(next_low_offset).to(self.device)

        # Precompute EEC table on GPU
        max_p2_slots = int(round(0.6 / self.dt))
        self._max_p2_slots = max_p2_slots
        eec_table = np.zeros((self._cycle_len, max_p2_slots + 1), dtype=np.float32)
        for start_mod in range(self._cycle_len):
            for L in range(max_p2_slots + 1):
                if L > 0:
                    idxs = (start_mod + np.arange(L)) % self._cycle_len
                    eec_table[start_mod, L] = (
                        float(price_cycle_np[idxs].sum()) * self.dt
                    )
        self._eec_table = torch.from_numpy(eec_table).to(self.device)

        # State tensors (initialized in reset)
        self.tasks: Optional[torch.Tensor] = None  # [batch, N, 5]
        self.current_times: Optional[torch.Tensor] = None  # [batch]
        self.twts: Optional[torch.Tensor] = None  # [batch]
        self.eecs: Optional[torch.Tensor] = None  # [batch]
        self.ws: Optional[torch.Tensor] = None  # [batch]
        self.masks: Optional[torch.Tensor] = None  # [batch, N]
        self.dones: Optional[torch.Tensor] = None  # [batch]
        self.remaining: Optional[torch.Tensor] = None  # [batch]

    def reset(
        self,
        tasks: Union[np.ndarray, torch.Tensor, None] = None,
        ws: Union[np.ndarray, torch.Tensor, None] = None,
        seed: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Reset with tasks/ws on GPU. Returns GPU tensors."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Handle tasks
        if tasks is not None:
            if isinstance(tasks, np.ndarray):
                self.tasks = torch.from_numpy(tasks).to(self.device)
            else:
                self.tasks = tasks.to(self.device)
        else:
            # Generate on CPU then move to GPU (generation is fast)
            from .data import generate_batch

            tasks_np = generate_batch(self.N, self.batch_size)
            self.tasks = torch.from_numpy(tasks_np).to(self.device)

        # Handle ws
        if ws is not None:
            if isinstance(ws, np.ndarray):
                self.ws = torch.from_numpy(ws).to(self.device)
            else:
                self.ws = ws.to(self.device)
        else:
            self.ws = (
                torch.rand(self.batch_size, device=self.device) * 0.98 + 0.01
            )  # [0.01, 0.99]

        actual_N = self.tasks.shape[1]

        # Initialize state tensors on GPU
        self.masks = torch.ones(self.batch_size, actual_N, device=self.device)
        self.remaining = torch.full(
            (self.batch_size,), actual_N, dtype=torch.int64, device=self.device
        )
        self.current_times = torch.zeros(self.batch_size, device=self.device)
        self.twts = torch.zeros(self.batch_size, device=self.device)
        self.eecs = torch.zeros(self.batch_size, device=self.device)
        self.dones = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        return self._get_obs()

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Get observation dict with GPU tensors."""
        # Compute EP vectorized
        current_slots = (self.current_times / self.dt).long()  # [batch]
        offsets = torch.arange(self.ep_horizon, device=self.device).unsqueeze(
            0
        )  # [1, H]
        slot_idxs = (
            current_slots.unsqueeze(1) + offsets
        ) % self._cycle_len  # [batch, H]
        EP = self._price_cycle[slot_idxs]  # [batch, H]

        return {
            "tasks": self.tasks,
            "EP": EP,
            "objs": torch.stack([self.twts, self.eecs], dim=1),  # [batch, 2]
            "w": self.ws.unsqueeze(1),  # [batch, 1]
            "mask": self.masks,  # [batch, N]
        }

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions. All inputs/outputs are GPU tensors.

        Args:
            actions: [batch] integer actions (already on GPU)

        Returns:
            obs, rewards, dones - all GPU tensors
        """
        rewards = torch.zeros(self.batch_size, device=self.device)

        active = ~self.dones
        if not active.any():
            return self._get_obs(), rewards, self.dones, {}

        # Parse actions
        task_idx = (actions // 2).long()  # [batch]
        mode = (actions % 2).long()  # [batch]

        # Gather task durations (only for active)
        batch_idx = torch.arange(self.batch_size, device=self.device)
        p1 = self.tasks[batch_idx, task_idx, 0]  # [batch]
        p2 = self.tasks[batch_idx, task_idx, 1]  # [batch]
        p3 = self.tasks[batch_idx, task_idx, 2]  # [batch]

        start_time = self.current_times
        step1_end = start_time + p1

        # Wait logic
        slot_after_step1 = (step1_end / self.dt).long()
        slot_mod = slot_after_step1 % self._cycle_len
        price_after = self._price_cycle[slot_mod]

        wait_time = torch.zeros(self.batch_size, device=self.device)
        need_wait = (mode == 1) & (price_after == 1.0) & active
        if need_wait.any():
            offsets = self._next_low_offset[slot_mod[need_wait]].float()
            time_to_low = offsets * self.dt
            wait_time[need_wait] = torch.minimum(
                time_to_low, torch.tensor(self.max_wait, device=self.device)
            )

        step2_start = step1_end + wait_time

        # Step2 EEC from lookup table
        start_slot2 = (step2_start / self.dt).long()
        start_mod2 = start_slot2 % self._cycle_len
        p2_slots = torch.round(p2 / self.dt).long().clamp(0, self._max_p2_slots)
        eec = self._eec_table[start_mod2, p2_slots]

        step3_end = step2_start + p2 + p3

        # Update state (only for active instances)
        self.current_times = torch.where(active, step3_end, self.current_times)
        self.twts = torch.where(active, self.twts + wait_time, self.twts)
        self.eecs = torch.where(active, self.eecs + eec, self.eecs)

        # Update masks using scatter
        self.masks[batch_idx[active], task_idx[active]] = 0.0
        self.remaining = torch.where(active, self.remaining - 1, self.remaining)

        # Check done
        done_now = (self.remaining <= 0) & active
        self.dones = self.dones | done_now

        # Compute rewards for newly done instances
        if done_now.any():
            final_eec = self.eecs[done_now] * 2
            rewards[done_now] = -torch.maximum(
                self.ws[done_now] * self.twts[done_now],
                (1.0 - self.ws[done_now]) * final_eec,
            )

        return self._get_obs(), rewards, self.dones, {}

    def get_final_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final (TWT, raw EEC) as GPU tensors."""
        return self.twts, self.eecs


if __name__ == "__main__":
    # Test environment
    print("Testing ECSPEnv...")

    env = ECSPEnv(N=5)
    obs, info = env.reset(seed=42, options={"w": 0.5})

    print(f"Initial observation:")
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print(f"Info: {info}")

    # Take random valid actions until done
    done = False
    step = 0
    while not done:
        valid = env.get_valid_actions()
        valid_indices = np.where(valid == 1)[0]
        action = np.random.choice(valid_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        print(
            f"Step {step}: action={action} (task={action//2}, mode={action%2}), "
            f"reward={reward:.4f}, done={done}"
        )

    twt, eec = env.get_final_metrics()
    print(f"\nFinal metrics: TWT={twt:.4f}, EEC={eec:.4f}")

    # Test batch environment
    print("\n\nTesting BatchECSPEnv...")
    batch_env = BatchECSPEnv(N=5, batch_size=4)
    obs = batch_env.reset(seed=42)

    print(f"Batch observation:")
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}")
