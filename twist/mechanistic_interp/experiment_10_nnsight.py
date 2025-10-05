#!/usr/bin/env python3
"""
Experiment #10: Think-Mode Circuit Discovery (NNsight Version)

Uses NNsight instead of TransformerLens - supports ANY HuggingFace model!
Including Qwen/Qwen2.5-14B, Qwen2.5-7B, or even your remote API model.

NNsight advantages:
- Works with any HuggingFace model (no limited model list)
- Can use remote models via API
- Similar intervention capabilities to TransformerLens

Usage:
  python experiment_10_nnsight.py --pair 1 --model Qwen/Qwen2.5-14B
  python experiment_10_nnsight.py --pair 1 --model Qwen/Qwen2.5-7B
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import OUTPUT_DIR_BASE_PATH

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ABLATION_DIR = OUTPUT_DIR_BASE_PATH / "ablation_outputs"
OUTPUT_DIR = Path("twist/mechanistic_interp/outputs/experiment_10")
CACHE_DIR = OUTPUT_DIR / "cache_nnsight"

# ALL 4 MATCHED PAIRS
MATCHED_PAIRS = [
    {
        "pair_id": 1,
        "baseline_id": "twist_001",
        "target_id": "twist_004",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
    },
    {
        "pair_id": 2,
        "baseline_id": "twist_001",
        "target_id": "twist_005",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
    },
    {
        "pair_id": 3,
        "baseline_id": "twist_004",
        "target_id": "twist_005",
        "clean_twist": "the world is a simulation",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["simulation", "glitch", "code", "fractal", "static"],
    },
    {
        "pair_id": 4,
        "baseline_id": "twist_005",
        "target_id": "twist_004",
        "clean_twist": "they are the last human alive",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["last human", "last person", "alone", "extinct"],
    }
]


def load_prompts(baseline_id: str, target_id: str, think_mode: str) -> str:
    """Load the ablated prompt."""
    rollout_file = (
        ABLATION_DIR / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )
    with open(rollout_file, 'r') as f:
        data = json.load(f)
    return data['ablated_prompt']


def compute_keyword_probability(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    keywords: List[str]
) -> float:
    """Compute average probability of keywords as next token."""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get probabilities at last position
    # Ensure tensor is on CPU and materialized before calling .item()
    probs = torch.softmax(logits[0, -1, :], dim=0).cpu()

    # Check keyword probabilities
    keyword_probs = []
    for keyword in keywords:
        # Tokenize keyword
        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        if len(keyword_tokens) > 0:
            first_token = keyword_tokens[0]
            # Ensure we have actual tensor before .item()
            prob_val = probs[first_token]
            if prob_val.device.type != 'meta':
                keyword_probs.append(prob_val.item())

    return np.mean(keyword_probs) if keyword_probs else 0.0


def patch_layer_nnsight(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str]
) -> Dict:
    """Layer-level patch using NNsight with proper tracing and length alignment."""

    # Baseline (no patching)
    baseline_prob = compute_keyword_probability(model, tokenizer, target_prompt, clean_keywords)

    # Tokenize both prompts
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(DEVICE)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)

    # Capture source layer output
    with model.trace(source_inputs) as t_src:
        src_o_proj = t_src(model.model.layers[layer].self_attn.o_proj).output.save()
        _ = model(**source_inputs)
    source_activation = src_o_proj.value

    # Patch into target with alignment
    with model.trace(target_inputs) as t_tgt:
        tgt_o_proj = t_tgt(model.model.layers[layer].self_attn.o_proj).output
        _ = model(**target_inputs)

        try:
            tgt_val = tgt_o_proj.value
        except Exception:
            _ = t_tgt(model.model.layers[layer].self_attn.o_proj).output.save()
            tgt_val = tgt_o_proj.value

        min_len = min(source_activation.shape[1], tgt_val.shape[1])
        tgt_o_proj[:, :min_len, :] = source_activation[:, :min_len, :]

        patched_logits = t_tgt(model.lm_head).output.save()

    # Compute patched probability
    # Ensure tensor is materialized on CPU before .item()
    probs = torch.softmax(patched_logits.value[0, -1, :], dim=0).cpu()

    keyword_probs = []
    for keyword in clean_keywords:
        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        if len(keyword_tokens) > 0:
            prob_val = probs[keyword_tokens[0]]
            if prob_val.device.type != 'meta':
                keyword_probs.append(prob_val.item())

    patched_prob = np.mean(keyword_probs) if keyword_probs else 0.0

    return {
        'layer': layer,
        'baseline_prob': float(baseline_prob),
        'patched_prob': float(patched_prob),
        'delta': float(patched_prob - baseline_prob)
    }


def scan_layer_nnsight(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str],
    cache_file: Path
) -> List[Dict]:
    """Scan all heads in a layer with caching."""

    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        return cached['results']

    # For now, we scan entire layer as one unit
    # (NNsight head-level patching requires knowing model architecture)
    # This is a simplified version - scans layer-level only

    results = []

    try:
        # Layer-level patching first (head-level requires arch-specific hooks)
        result = patch_layer_nnsight(
            model, tokenizer, source_prompt, target_prompt, layer, clean_keywords
        )
        result['head'] = 'all'  # Mark as layer-level
        results.append(result)

    except Exception as e:
        print(f"\n  Error at layer {layer}: {e}")

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({'layer': layer, 'results': results}, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment #10 with NNsight")
    parser.add_argument('--pair', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                       help='Any HuggingFace model (e.g., Qwen/Qwen2.5-14B, Qwen/Qwen2.5-7B)')
    args = parser.parse_args()

    config = MATCHED_PAIRS[args.pair - 1]

    print("=" * 80)
    print(f"EXPERIMENT #10: NNsight Version (Pair {args.pair}/4)")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Pair: {config['baseline_id']} → {config['target_id']}")

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

    print(f"Source: {len(source_prompt)} chars")
    print(f"Target: {len(target_prompt)} chars")

    # Load model with NNsight
    print(f"\nLoading model with NNsight: {args.model}")
    print("This may take a few minutes...")

    # Load model on specific device (not 'auto' to avoid meta tensors)
    model = LanguageModel(
        args.model,
        device_map=DEVICE,  # Use specific device, not 'auto'
        torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"✓ Model loaded")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")

    # Scan layers
    print("\n" + "=" * 80)
    print("SCANNING LAYERS (Layer-level patching)")
    print("=" * 80)
    print("\nNote: This version does layer-level patching (faster)")
    print("For head-level, we need architecture-specific code")

    all_results = []
    n_layers = model.config.num_hidden_layers

    for layer in tqdm(range(n_layers), desc="Layers"):
        cache_file = CACHE_DIR / f"pair{args.pair}_layer{layer:02d}.json"

        layer_results = scan_layer_nnsight(
            model, tokenizer, source_prompt, target_prompt,
            layer, config['clean_keywords'], cache_file
        )

        all_results.extend(layer_results)

    # Sort and save
    all_results_sorted = sorted(all_results, key=lambda x: x.get('delta', 0.0), reverse=True)

    output_file = OUTPUT_DIR / f"pair{args.pair}_nnsight_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'pair_config': config,
            'model': args.model,
            'total_layers': n_layers,
            'results': all_results_sorted,
            'top_10': all_results_sorted[:10]
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Display
    print("\n" + "=" * 80)
    print("TOP 10 LAYERS")
    print("=" * 80)
    print(f"{'Layer':<8} {'Baseline':<12} {'Patched':<12} {'Delta':<12}")
    print("-" * 80)

    for result in all_results_sorted[:10]:
        print(
            f"{result['layer']:<8} "
            f"{result['baseline_prob']:<12.6f} "
            f"{result['patched_prob']:<12.6f} "
            f"{result['delta']:+12.6f}"
        )


if __name__ == "__main__":
    main()
