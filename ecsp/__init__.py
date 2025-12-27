"""
ECSP - Energy-Conscious Scheduling Problem Package.
Paper-exact implementation of ECSPNet.

VERSION: 2.1-GPU-ASYNC - Full GPU acceleration with async data prefetch
"""

from .data import (
    generate_instance,
    generate_batch,
    generate_tou_pattern,
    get_price_at_slot,
    compute_total_processing_time,
    BENCHMARK_SCALES,
    TRAINING_CONFIG,
    INFERENCE_CONFIG,
    ENV_CONFIG,
    DT,
)

from .env import ECSPEnv, BatchECSPEnv, GPUBatchECSPEnv, ENV_VERSION

from .model import ECSPNet, obs_dict_to_tensors

from .train import Trainer, train_model, TRAIN_VERSION

from .infer import (
    Inferencer,
    Solution,
    compute_pareto_front,
    compute_hypervolume,
    load_model_for_inference,
    evaluate_on_benchmark,
    visualize_pareto_front,
)

__version__ = "2.1.0-GPU-ASYNC"
print(f"[ECSP Package v{__version__}] Loaded with GPU + async prefetch")

__all__ = [
    # Data
    "generate_instance",
    "generate_batch",
    "generate_tou_pattern",
    "get_price_at_slot",
    "compute_total_processing_time",
    "BENCHMARK_SCALES",
    "TRAINING_CONFIG",
    "INFERENCE_CONFIG",
    "ENV_CONFIG",
    "DT",
    # Environment
    "ECSPEnv",
    "BatchECSPEnv",
    "GPUBatchECSPEnv",
    "ENV_VERSION",
    # Model
    "ECSPNet",
    "obs_dict_to_tensors",
    # Training
    "Trainer",
    "train_model",
    # Inference
    "Inferencer",
    "Solution",
    "compute_pareto_front",
    "compute_hypervolume",
    "load_model_for_inference",
    "evaluate_on_benchmark",
    "visualize_pareto_front",
]
