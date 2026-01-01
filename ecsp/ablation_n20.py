"""
Ablation Study: Tchebycheff Decomposition vs Direct Weighted Addition
Following Table IV in the paper.

IMPORTANT: This ablation compares models TRAINED with different scalarizations.
The difference is in the TRAINING reward, not inference.

- Tchebycheff: R = -max(w*TWT, (1-w)*EEC)  [main paper method]
- Weighted:    R = -(w*TWT + (1-w)*EEC)    [ablation comparison]

To properly reproduce Table IV, you need TWO separately trained models:
1. The main model (trained with Tchebycheff - your current model)
2. An ablation model (trained with weighted addition)

If only the Tchebycheff model is provided, this script will note that
the comparison cannot be fully reproduced.
"""

import os
import json
import time
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import warnings

from .data import generate_instance, TRAINING_CONFIG, ENV_CONFIG
from .env import ECSPEnv
from .model import ECSPNet
from .evaluate_n20 import (
    N, B_SOLUTIONS, BETA, HV_REFERENCE, NUM_TEST_CASES,
    generate_test_cases, get_pareto_front, compute_hypervolume,
    ecspnet_inference, wilcoxon_test,
    Solution, ParetoFront
)


def run_ablation_study(
    model_tchebycheff_path: str,
    model_weighted_path: Optional[str] = None,
    output_dir: str = "evaluation_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run decomposition ablation study (Table IV).
    
    Compares models TRAINED with:
    - Tchebycheff decomposition: R = -max(w*TWT, (1-w)*EEC)
    - Direct weighted addition: R = -(w*TWT + (1-w)*EEC)
    
    NOTE: The difference is in TRAINING, not inference. Both models
    use the same inference procedure (B=1000, β=0.9).
    
    Args:
        model_tchebycheff_path: Path to model trained with Tchebycheff (required)
        model_weighted_path: Path to model trained with weighted addition (optional)
        output_dir: Output directory
        device: Torch device
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    print("=" * 60)
    print("Ablation Study: Tchebycheff vs Weighted Addition")
    print("=" * 60)
    print("\nThis tests the impact of different scalarizations during TRAINING.")
    print("Both models use identical inference (B=1000, β=0.9).\n")
    
    # Load Tchebycheff model (required)
    print("1. Loading Tchebycheff model...")
    model_tcheb = ECSPNet(
        d_model=TRAINING_CONFIG['d_model'],
        num_heads=TRAINING_CONFIG['num_heads'],
        num_blocks=TRAINING_CONFIG['num_blocks'],
    ).to(device)
    
    checkpoint = torch.load(model_tchebycheff_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model_tcheb.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_tcheb.load_state_dict(checkpoint)
    model_tcheb.eval()
    print(f"   Loaded from {model_tchebycheff_path}")
    
    # Check for weighted model
    has_weighted_model = False
    if model_weighted_path and os.path.exists(model_weighted_path):
        print("\n2. Loading Weighted Addition model...")
        model_weighted = ECSPNet(
            d_model=TRAINING_CONFIG['d_model'],
            num_heads=TRAINING_CONFIG['num_heads'],
            num_blocks=TRAINING_CONFIG['num_blocks'],
        ).to(device)
        checkpoint = torch.load(model_weighted_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model_weighted.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_weighted.load_state_dict(checkpoint)
        model_weighted.eval()
        has_weighted_model = True
        print(f"   Loaded from {model_weighted_path}")
    else:
        print("\n2. Weighted Addition model NOT provided.")
        print("   ⚠️  To fully reproduce Table IV, you need to:")
        print("      1. Modify train.py to use weighted addition reward")
        print("      2. Train a separate model")
        print("      3. Provide it via --model-weighted argument")
        print("\n   This run will only show Tchebycheff results.")
        model_weighted = None
    
    # Load test cases
    print("\n3. Loading test cases...")
    test_cases = generate_test_cases(os.path.join(output_dir, "test_cases"))
    
    results = {
        'Tchebycheff': {'hv': [], 'time': []},
    }
    if has_weighted_model:
        results['Weighted'] = {'hv': [], 'time': []}
    
    # Evaluate
    for case_idx, tasks in enumerate(test_cases):
        print(f"\n4.{case_idx + 1}. Evaluating Case {case_idx + 1}...")
        
        # Tchebycheff model
        print("   - Tchebycheff model...")
        pf_tcheb, time_tcheb = ecspnet_inference(model_tcheb, tasks, device)
        front_tcheb = pf_tcheb.to_array()
        hv_tcheb = compute_hypervolume(front_tcheb)
        results['Tchebycheff']['hv'].append(hv_tcheb)
        results['Tchebycheff']['time'].append(time_tcheb)
        print(f"     HV={hv_tcheb:.4f}, Time={time_tcheb:.2f}s")
        
        # Weighted model (if available)
        if has_weighted_model:
            print("   - Weighted Addition model...")
            pf_weighted, time_weighted = ecspnet_inference(model_weighted, tasks, device)
            front_weighted = pf_weighted.to_array()
            hv_weighted = compute_hypervolume(front_weighted)
            results['Weighted']['hv'].append(hv_weighted)
            results['Weighted']['time'].append(time_weighted)
            print(f"     HV={hv_weighted:.4f}, Time={time_weighted:.2f}s")
    
    # Generate ablation table (Table IV style)
    print("\n5. Generating ablation table...")
    
    lines = [
        "# Ablation Study: Decomposition Method (N=20)",
        "",
        "This compares models TRAINED with different scalarizations.",
        "",
        "| Method | Case 1 | Case 2 | Case 3 | Mean | Sig |",
        "|--------|--------|--------|--------|------|-----|",
    ]
    
    # Tchebycheff row
    vals = results['Tchebycheff']['hv']
    mean_val = np.mean(vals)
    lines.append(f"| Tchebycheff | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {mean_val:.4f} | - |")
    
    # Weighted row (if available)
    if has_weighted_model:
        vals_w = results['Weighted']['hv']
        mean_val_w = np.mean(vals_w)
        _, sig = wilcoxon_test(results['Tchebycheff']['hv'], results['Weighted']['hv'])
        lines.append(f"| Weighted | {vals_w[0]:.4f} | {vals_w[1]:.4f} | {vals_w[2]:.4f} | {mean_val_w:.4f} | {sig} |")
        
        # Improvement row
        improvement = [
            results['Tchebycheff']['hv'][i] - results['Weighted']['hv'][i]
            for i in range(NUM_TEST_CASES)
        ]
        mean_imp = np.mean(improvement)
        lines.append("")
        lines.append(f"Tchebycheff improvement: {mean_imp:+.4f} mean HV")
    else:
        lines.append("| Weighted | N/A | N/A | N/A | N/A | - |")
        lines.append("")
        lines.append("⚠️ Weighted Addition model not provided - ablation incomplete.")
        lines.append("Train a model with R = -(w*TWT + (1-w)*EEC) reward.")
    
    lines.append("")
    lines.append("Sig: + = Tchebycheff better, - = Weighted better, = = no difference")
    
    ablation_table = "\n".join(lines)
    
    with open(os.path.join(output_dir, "table_ablation.md"), 'w') as f:
        f.write(ablation_table)
    
    print(f"\n{ablation_table}")
    print(f"\nSaved to {os.path.join(output_dir, 'table_ablation.md')}")
    
    # Save JSON results
    with open(os.path.join(output_dir, "ablation_results.json"), 'w') as f:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()} 
                   for k, v in results.items()}, f, indent=2)
    
    return results


def print_training_instructions():
    """Print instructions for training a weighted addition model."""
    print("""
================================================================================
How to Train a Weighted Addition Model for Ablation
================================================================================

To properly reproduce Table IV, you need to train a second model using
weighted addition instead of Tchebycheff decomposition.

Modify the reward computation in train.py:

CURRENT (Tchebycheff):
    reward = -torch.max(w * twt, (1 - w) * eec)

CHANGE TO (Weighted Addition):
    reward = -(w * twt + (1 - w) * eec)

Then train:
    python -m ecsp.main train --scales 20 --epochs 3000 --device cuda

Save the checkpoint and provide it to this script:
    python -m ecsp.ablation_n20 --model-tcheb checkpoints/ecspnet_N20_final.pt \\
                                 --model-weighted checkpoints/ecspnet_N20_weighted.pt
================================================================================
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation study: Tchebycheff vs Weighted")
    parser.add_argument("--model-tcheb", type=str, default="checkpoints/ecspnet_N20_final.pt",
                        help="Path to Tchebycheff-trained model (required)")
    parser.add_argument("--model-weighted", type=str, default=None,
                        help="Path to weighted-addition-trained model (optional)")
    parser.add_argument("--output", type=str, default="evaluation_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--instructions", action="store_true",
                        help="Print instructions for training weighted model")
    
    args = parser.parse_args()
    
    if args.instructions:
        print_training_instructions()
    else:
        run_ablation_study(args.model_tcheb, args.model_weighted, args.output, args.device)
