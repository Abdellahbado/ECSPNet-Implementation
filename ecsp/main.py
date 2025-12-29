"""
Main entry point for ECSPNet training and evaluation.
Trains models on benchmark scales N = [20, 40, 60, 100].

VERSION: 2.3-AUTO-RESUME - Automatic checkpoint resume for Kaggle
"""

import argparse
import os
import re
import glob
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ecsp.data import BENCHMARK_SCALES, TRAINING_CONFIG, generate_instance
from ecsp.model import ECSPNet
from ecsp.train import Trainer, train_model, TRAIN_VERSION
from ecsp.infer import (
    Inferencer,
    load_model_for_inference,
    evaluate_on_benchmark,
    visualize_pareto_front,
)
from ecsp import __version__ as PACKAGE_VERSION

MAIN_VERSION = "2.3-AUTO-RESUME"


def find_latest_checkpoint(save_dir: str, N: int) -> Optional[Tuple[str, int]]:
    """
    Find the latest checkpoint for a given N.

    Args:
        save_dir: Directory containing checkpoints
        N: Number of tasks

    Returns:
        Tuple of (checkpoint_path, epoch) or None if no checkpoint found
    """
    if not os.path.exists(save_dir):
        return None

    # Pattern: ecspnet_N{N}_epoch{epoch}.pt
    pattern = os.path.join(save_dir, f"ecspnet_N{N}_epoch*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        # Also check for final checkpoint
        final_path = os.path.join(save_dir, f"ecspnet_N{N}_final.pt")
        if os.path.exists(final_path):
            try:
                ckpt = torch.load(final_path, map_location="cpu")
                return (final_path, ckpt.get("epoch", 0))
            except:
                pass
        return None

    # Extract epoch numbers and find max
    latest_epoch = -1
    latest_path = None

    for ckpt_path in checkpoints:
        match = re.search(r"epoch(\d+)\.pt$", ckpt_path)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = ckpt_path

    if latest_path:
        return (latest_path, latest_epoch)
    return None


def train_all_scales(
    scales: List[int] = BENCHMARK_SCALES,
    num_epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    device: str = "cuda",
    save_dir: str = "checkpoints",
    auto_resume: bool = True,
):
    """
    Train models for all benchmark scales with automatic checkpoint resume.

    Args:
        scales: List of N values to train on
        num_epochs: Training epochs per scale
        batch_size: Batch size
        device: Training device
        save_dir: Directory to save checkpoints
        auto_resume: Automatically resume from latest checkpoint if found
    """
    print("=" * 60)
    print(f"ECSPNet Training v{MAIN_VERSION} - AUTO RESUME ENABLED")
    print(f"Package: {PACKAGE_VERSION}, Trainer: {TRAIN_VERSION}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Async prefetch: ENABLED (4 CPU workers)")
    else:
        print("WARNING: No GPU available, using CPU")
    print("=" * 60)

    results = {}

    for N in scales:
        print(f"\n{'=' * 60}")
        print(f"Training for N = {N}")
        print(f"{'=' * 60}")

        # Check for existing checkpoint
        resume_from = None
        if auto_resume:
            checkpoint_info = find_latest_checkpoint(save_dir, N)
            if checkpoint_info:
                resume_from, last_epoch = checkpoint_info
                print(f"[AUTO-RESUME] Found checkpoint at epoch {last_epoch}")
                print(f"[AUTO-RESUME] Path: {resume_from}")

                # Skip if already completed
                if last_epoch >= num_epochs:
                    print(
                        f"[AUTO-RESUME] Training already complete for N={N}, skipping..."
                    )
                    continue
            else:
                print(f"[AUTO-RESUME] No checkpoint found, starting fresh")

        trainer = Trainer(
            N=N,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            save_dir=save_dir,
        )

        trainer.train(resume_from=resume_from)
        trainer.save_history()

        # Record final metrics
        results[N] = {
            "final_loss": trainer.history["loss"][-1],
            "final_reward": trainer.history["mean_reward"][-1],
            "final_twt": trainer.history["mean_twt"][-1],
            "final_eec": trainer.history["mean_eec"][-1],
        }

    # Save overall results
    results_path = os.path.join(save_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 60}")

    return results


def evaluate_all_scales(
    scales: List[int] = BENCHMARK_SCALES,
    checkpoint_dir: str = "checkpoints",
    num_test_instances: int = 100,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Evaluate trained models on all benchmark scales.

    Args:
        scales: List of N values to evaluate
        checkpoint_dir: Directory with trained checkpoints
        num_test_instances: Number of test instances per scale
        device: Inference device
        seed: Random seed
    """
    print("=" * 60)
    print("ECSPNet Evaluation - Paper Exact Implementation")
    print("=" * 60)

    all_results = {}

    for N in scales:
        checkpoint_path = os.path.join(checkpoint_dir, f"ecspnet_N{N}_final.pt")

        if not os.path.exists(checkpoint_path):
            print(f"\nSkipping N={N}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating N = {N}")
        print(f"{'=' * 60}")

        # Load model
        model, config = load_model_for_inference(checkpoint_path, device)

        # Evaluate
        results = evaluate_on_benchmark(
            model=model,
            N=N,
            num_instances=num_test_instances,
            device=device,
            seed=seed,
            verbose=True,
        )

        all_results[N] = results

    # Save evaluation results
    results_path = os.path.join(checkpoint_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 60}")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(
        f"{'N':>6} | {'PF Size':>10} | {'Mean TWT':>12} | {'Mean EEC':>12} | {'HV':>12}"
    )
    print("-" * 80)
    for N, res in all_results.items():
        print(
            f"{N:>6} | {res['mean_pf_size']:>10.2f} | {res['mean_twt']:>12.4f} | "
            f"{res['mean_eec']:>12.4f} | {res['mean_hypervolume']:>12.4f}"
        )
    print("-" * 80)

    return all_results


def demo_single_instance(
    N: int = 20,
    checkpoint_path: str = None,
    device: str = "cpu",
    seed: int = 42,
    visualize: bool = True,
):
    """
    Demo solving a single instance and visualizing the Pareto front.

    Args:
        N: Number of tasks
        checkpoint_path: Path to trained model (uses random model if None)
        device: Inference device
        seed: Random seed
        visualize: Whether to visualize results
    """
    print("=" * 60)
    print(f"ECSPNet Demo - Single Instance (N={N})")
    print("=" * 60)

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load or create model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, config = load_model_for_inference(checkpoint_path, device)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Using untrained model (for demo purposes)")
        model = ECSPNet()
        model.eval()

    # Generate instance
    tasks = generate_instance(N, seed=seed)
    print(f"\nGenerated instance with {N} tasks")
    print(f"Task features (first 5):")
    for i in range(min(5, N)):
        print(
            f"  Task {i}: p1={tasks[i,0]:.1f}, p2={tasks[i,1]:.1f}, "
            f"p3={tasks[i,2]:.1f}"
        )

    # Create inferencer
    inferencer = Inferencer(model, torch.device(device), B=100)  # Reduced B for demo

    # Solve instance
    print("\nSolving instance...")
    pf, all_sols = inferencer.solve_instance(tasks, return_all=True, verbose=True)

    # Print results
    print(f"\nResults:")
    print(f"  Total solutions generated: {len(all_sols)}")
    print(f"  Pareto front size: {len(pf)}")

    print(f"\nPareto front (first 10):")
    print(f"{'#':>3} | {'TWT':>10} | {'EEC':>10} | {'w':>8}")
    print("-" * 40)
    for i, sol in enumerate(pf[:10]):
        print(f"{i+1:>3} | {sol.twt:>10.4f} | {sol.eec:>10.4f} | {sol.w:>8.3f}")
    if len(pf) > 10:
        print(f"... and {len(pf) - 10} more solutions")

    # Visualize
    if visualize:
        try:
            visualize_pareto_front(
                pf,
                all_sols,
                title=f"Pareto Front (N={N})",
                save_path=f"pareto_front_N{N}.png",
            )
        except Exception as e:
            print(f"Visualization skipped: {e}")

    return pf, all_sols


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="ECSPNet - Energy-Conscious Scheduling Policy Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all benchmark scales
  python main.py train --epochs 3000 --device cuda
  
  # Train on specific scale
  python main.py train --scales 20 40 --epochs 1000
  
  # Evaluate trained models
  python main.py eval --checkpoint-dir checkpoints
  
  # Demo single instance
  python main.py demo --N 20 --visualize
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=BENCHMARK_SCALES,
        help="Task scales to train on (default: 20 40 60 100)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["epochs"],
        help="Number of training epochs (default: 3000)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size (default: 2048)",
    )
    train_parser.add_argument(
        "--device", type=str, default="cuda", help="Device to train on (default: cuda)"
    )
    train_parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained models")
    eval_parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=BENCHMARK_SCALES,
        help="Task scales to evaluate",
    )
    eval_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory with trained checkpoints",
    )
    eval_parser.add_argument(
        "--num-instances",
        type=int,
        default=100,
        help="Number of test instances per scale",
    )
    eval_parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference"
    )
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Demo on single instance")
    demo_parser.add_argument("--N", type=int, default=20, help="Number of tasks")
    demo_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (uses random model if not provided)",
    )
    demo_parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference"
    )
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    demo_parser.add_argument(
        "--no-visualize", action="store_true", help="Skip visualization"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_all_scales(
            scales=args.scales,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
        )

    elif args.command == "eval":
        evaluate_all_scales(
            scales=args.scales,
            checkpoint_dir=args.checkpoint_dir,
            num_test_instances=args.num_instances,
            device=args.device,
            seed=args.seed,
        )

    elif args.command == "demo":
        demo_single_instance(
            N=args.N,
            checkpoint_path=args.checkpoint,
            device=args.device,
            seed=args.seed,
            visualize=not args.no_visualize,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
