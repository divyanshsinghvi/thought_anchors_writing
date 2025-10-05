#!/usr/bin/env python3
"""
Experiment #10: Think-Mode Circuit Discovery (Multi-Pair, Cached, Resumable)

Research Question:
Why does allow_more_thinking lead to FULL_OVERRIDE (78%) while no_more_thinking
ALWAYS produces BLENDED outputs?

Strategy:
1. Run on 4 matched pairs (embedded below - start with pair 1)
2. Cache results per layer (resumable if crashes)
3. Patch heads from no_more_thinking → allow_more_thinking
4. Measure which heads restore clean twist signal

Matched Pairs (All 4 embedded in MATCHED_PAIRS constant):
1. twist_001 → twist_004: dream vs simulation
2. twist_001 → twist_005: dream vs last human
3. twist_004 → twist_005: simulation vs last human
4. twist_005 → twist_004: last human vs simulation

Usage:
  python experiment_10_think_mode_circuit.py --pair 1
  python experiment_10_think_mode_circuit.py --pair 2  # validate on another pair

Results are cached per layer in outputs/experiment_10/cache/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import OUTPUT_DIR_BASE_PATH

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ABLATION_DIR = OUTPUT_DIR_BASE_PATH / "ablation_outputs"
OUTPUT_DIR = Path("twist/mechanistic_interp/outputs/experiment_10")
CACHE_DIR = OUTPUT_DIR / "cache"

# ALL 4 MATCHED PAIRS EMBEDDED HERE
MATCHED_PAIRS = [
    {
        "pair_id": 1,
        "baseline_id": "twist_001",
        "target_id": "twist_004",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
        "corrupted_keywords": ["simulation", "glitch", "code", "fractal", "static"]
    },
    {
        "pair_id": 2,
        "baseline_id": "twist_001",
        "target_id": "twist_005",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
        "corrupted_keywords": ["last human", "last person", "alone", "extinct", "survivor"]
    },
    {
        "pair_id": 3,
        "baseline_id": "twist_004",
        "target_id": "twist_005",
        "clean_twist": "the world is a simulation",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["simulation", "glitch", "code", "fractal", "static"],
        "corrupted_keywords": ["last human", "last person", "alone", "extinct", "survivor"]
    },
    {
        "pair_id": 4,
        "baseline_id": "twist_005",
        "target_id": "twist_004",
        "clean_twist": "they are the last human alive",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["last human", "last person", "alone", "extinct", "survivor"],
        "corrupted_keywords": ["simulation", "glitch", "code", "fractal", "static"]
    }
]


def load_prompts(baseline_id: str, target_id: str, think_mode: str) -> str:
    """Load the ablated prompt for a specific configuration."""
    rollout_file = (
        ABLATION_DIR / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )

    with open(rollout_file, 'r') as f:
        data = json.load(f)

    return data['ablated_prompt']


def compute_keyword_probability(
    model: HookedTransformer,
    prompt: str,
    keywords: List[str]
) -> float:
    """
    Compute average probability of keywords appearing as next token.
    Simple proxy for whether model will generate clean twist.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    with torch.no_grad():
        logits = model(tokens)

    # Get probabilities at last position
    probs = torch.softmax(logits[0, -1, :], dim=0)

    # Check probabilities for keywords
    keyword_probs = []
    for keyword in keywords:
        keyword_tokens = model.to_tokens(keyword, prepend_bos=False)[0]
        if len(keyword_tokens) > 0:
            first_token = keyword_tokens[0].item()
            keyword_probs.append(probs[first_token].item())

    return np.mean(keyword_probs) if keyword_probs else 0.0


def patch_single_head(
    model: HookedTransformer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    head: int,
    clean_keywords: List[str]
) -> Dict:
    """Patch a single head and measure effect."""

    # Baseline (no patching)
    baseline_prob = compute_keyword_probability(model, target_prompt, clean_keywords)

    # Get source activation
    source_tokens = model.to_tokens(source_prompt, prepend_bos=True)
    target_tokens = model.to_tokens(target_prompt, prepend_bos=True)

    source_cache = {}

    def cache_hook(activation, hook):
        source_cache[hook.name] = activation.clone()
        return activation

    hook_name = tl_utils.get_act_name("z", layer)
    model.add_hook(hook_name, cache_hook)

    with torch.no_grad():
        _ = model(source_tokens)

    model.reset_hooks()

    # Patch this head
    def patch_hook(activation, hook):
        source_activation = source_cache[hook.name]
        # Patch all positions for this head
        min_len = min(activation.shape[1], source_activation.shape[1])
        activation[0, :min_len, head, :] = source_activation[0, :min_len, head, :]
        return activation

    model.add_hook(hook_name, patch_hook)

    patched_prob = compute_keyword_probability(model, target_prompt, clean_keywords)

    model.reset_hooks()

    return {
        'layer': layer,
        'head': head,
        'baseline_prob': baseline_prob,
        'patched_prob': patched_prob,
        'delta': patched_prob - baseline_prob
    }


def scan_layer(
    model: HookedTransformer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str],
    cache_file: Path
) -> List[Dict]:
    """Scan all heads in a single layer with caching."""

    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        return cached['results']

    # Run scan
    n_heads = model.cfg.n_heads
    results = []

    for head in range(n_heads):
        try:
            result = patch_single_head(
                model, source_prompt, target_prompt, layer, head, clean_keywords
            )
            results.append(result)
        except Exception as e:
            print(f"\n  Error at layer {layer} head {head}: {e}")
            continue

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({'layer': layer, 'results': results}, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment #10 on a specific pair")
    parser.add_argument(
        '--pair', type=int, default=1, choices=[1, 2, 3, 4],
        help='Which pair to run (1-4)'
    )
    parser.add_argument(
        '--model', type=str, default='Qwen/QwQ-32B-Preview',
        help='Model to use (default: Qwen/QwQ-32B-Preview, a reasoning-focused model)'
    )
    args = parser.parse_args()

    config = MATCHED_PAIRS[args.pair - 1]

    print("=" * 80)
    print(f"EXPERIMENT #10: Think-Mode Circuit (Pair {args.pair}/4)")
    print("=" * 80)

    print(f"\nPair {config['pair_id']}: {config['baseline_id']} → {config['target_id']}")
    print(f"Clean twist: {config['clean_twist']}")
    print(f"Corrupted twist: {config['corrupted_twist']}")

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print("\nLoading prompts...")
    source_prompt = load_prompts(
        config['baseline_id'], config['target_id'], 'no_more_thinking'
    )
    target_prompt = load_prompts(
        config['baseline_id'], config['target_id'], 'allow_more_thinking'
    )

    print(f"Source (no_more_thinking): {len(source_prompt)} chars")
    print(f"Target (allow_more_thinking): {len(target_prompt)} chars")

    # Load model
    print(f"\nLoading model: {args.model}")

    # Estimate memory requirements
    memory_estimates = {
        'Qwen/QwQ-32B-Preview': '~64GB',
        'Qwen/Qwen2.5-14B-Instruct': '~28GB',
        'Qwen/Qwen2.5-7B-Instruct': '~14GB',
        'meta-llama/Meta-Llama-3-8B-Instruct': '~16GB'
    }
    memory_est = memory_estimates.get(args.model, '~unknown')

    print(f"WARNING: This requires {memory_est} GPU memory")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    from twist.mechanistic_interp.setup import load_model
    model = load_model(args.model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print(f"\nModel: {n_layers} layers × {n_heads} heads = {n_layers * n_heads} total heads")

    # Scan all layers (with caching)
    print("\n" + "=" * 80)
    print("SCANNING ALL HEADS (Cached per layer)")
    print("=" * 80)

    all_results = []

    for layer in tqdm(range(n_layers), desc="Layers"):
        cache_file = CACHE_DIR / f"pair{args.pair}_layer{layer:02d}.json"

        layer_results = scan_layer(
            model, source_prompt, target_prompt, layer,
            config['clean_keywords'], cache_file
        )

        all_results.extend(layer_results)

        # Show progress
        if (layer + 1) % 10 == 0:
            tqdm.write(f"  Completed {layer + 1}/{n_layers} layers (cache: {cache_file.name})")

    # Sort by delta
    all_results_sorted = sorted(all_results, key=lambda x: x['delta'], reverse=True)

    # Save final results
    output_file = OUTPUT_DIR / f"pair{args.pair}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'pair_config': config,
            'total_heads': len(all_results),
            'results': all_results_sorted,
            'top_20': all_results_sorted[:20]
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Display top results
    print("\n" + "=" * 80)
    print("TOP 20 HEADS")
    print("=" * 80)
    print(f"{'Layer':<8} {'Head':<8} {'Baseline':<12} {'Patched':<12} {'Delta':<12}")
    print("-" * 80)

    for result in all_results_sorted[:20]:
        print(
            f"{result['layer']:<8} "
            f"{result['head']:<8} "
            f"{result['baseline_prob']:<12.6f} "
            f"{result['patched_prob']:<12.6f} "
            f"{result['delta']:+12.6f}"
        )

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run on other pairs to validate: --pair 2, --pair 3, --pair 4")
    print("2. Visualize attention for top heads")
    print("3. Generate actual text with top heads patched")


if __name__ == "__main__":
    main()
