"""
ECSP - Energy-Conscious Scheduling Problem Package.
Paper-exact implementation of ECSPNet.
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

from .env import ECSPEnv, BatchECSPEnv

from .model import ECSPNet, obs_dict_to_tensors

from .train import Trainer, train_model

from .infer import (
    Inferencer,
    Solution,
    compute_pareto_front,
    compute_hypervolume,
    load_model_for_inference,
    evaluate_on_benchmark,
    visualize_pareto_front,
)

__version__ = "1.0.0"
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
