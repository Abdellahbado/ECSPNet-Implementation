"""
Run complete paper-exact evaluation for N=20.

Usage:
    python -m ecsp.run_evaluation [--full] [--ablation] [--device cuda]
    python -m ecsp.run_evaluation --ablation --model-weighted checkpoints/weighted.pt
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run paper-exact evaluation for N=20"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full evaluation with all baselines"
    )
    parser.add_argument(
        "--ablation", action="store_true", 
        help="Run ablation study (Tchebycheff vs Weighted)"
    )
    parser.add_argument(
        "--model", type=str, default="checkpoints/ecspnet_N20_final.pt",
        help="Path to trained model checkpoint (Tchebycheff)"
    )
    parser.add_argument(
        "--model-weighted", type=str, default=None,
        help="Path to weighted-addition trained model (for ablation)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Default: run both
    if not args.full and not args.ablation:
        args.full = True
        args.ablation = True
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first or specify correct path with --model")
        sys.exit(1)
    
    # Run full evaluation
    if args.full:
        print("\n" + "=" * 60)
        print("Running Full Evaluation")
        print("=" * 60)
        from .evaluate_n20 import run_full_evaluation
        run_full_evaluation(args.model, args.output, args.device)
    
    # Run ablation
    if args.ablation:
        print("\n" + "=" * 60)
        print("Running Ablation Study")
        print("=" * 60)
        from .ablation_n20 import run_ablation_study
        run_ablation_study(args.model, args.model_weighted, args.output, args.device)
    
    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print(f"Results saved to: {args.output}/")
    print("=" * 60)
    
    # Print summary of output files
    print("\nOutput files:")
    if os.path.exists(args.output):
        for f in sorted(os.listdir(args.output)):
            fpath = os.path.join(args.output, f)
            if os.path.isfile(fpath):
                print(f"  - {f}")


if __name__ == "__main__":
    main()
