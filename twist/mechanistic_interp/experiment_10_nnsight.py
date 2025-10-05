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

def get_resid_handle(model, layer: int, location: str):
    """Return traced handle for residual location: pre_attn, post_attn, post_mlp."""
    if location == "pre_attn":
        return model.model.layers[layer].self_attn.input[0][0]
    if location == "post_attn":
        return model.model.layers[layer].self_attn.output
    if location == "post_mlp":
        return model.model.layers[layer].output
    raise ValueError("location must be one of: pre_attn, post_attn, post_mlp")


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
    """Compute average probability of keywords as next token using NNsight trace."""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Use NNsight trace to get logits
    with model.trace(inputs):
        logits_out = model.output.save()

    # Get probabilities at last position - move to CPU immediately
    logits = logits_out.logits.cpu()
    probs = torch.softmax(logits[0, -1, :], dim=0)

    # Check keyword probabilities
    keyword_probs = []
    for keyword in keywords:
        # Tokenize keyword
        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        if len(keyword_tokens) > 0:
            first_token = keyword_tokens[0]
            prob_val = probs[first_token]
            keyword_probs.append(prob_val.item())

    result = np.mean(keyword_probs) if keyword_probs else 0.0

    # Free memory
    del logits, probs, logits_out, inputs
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return result


def patch_layer_nnsight(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str],
    corrupted_keywords: List[str],
    location: str = "post_attn"
) -> Dict:
    """Layer-level patch using NNsight with proper tracing and length alignment."""

    # Baseline (no patching)
    baseline_clean = compute_keyword_probability(model, tokenizer, target_prompt, clean_keywords)
    baseline_corr = compute_keyword_probability(model, tokenizer, target_prompt, corrupted_keywords)

    # Tokenize both prompts
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(DEVICE)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)

    # Capture source residual at chosen location
    with model.trace(source_inputs):
        src_handle = get_resid_handle(model, layer, location).save()
    source_activation = src_handle.cpu()  # Move to CPU immediately

    # Patch into target with alignment
    with model.trace(target_inputs):
        tgt_handle = get_resid_handle(model, layer, location)
        tgt_len = int(target_inputs["input_ids"].shape[1])
        min_len = min(source_activation.shape[1], tgt_len)
        # Move source back to device for patching
        tgt_handle[:, :min_len, :] = source_activation[:, :min_len, :].to(DEVICE)
        traced_out = model.output.save()

    # Compute patched probabilities - move to CPU immediately
    logits = traced_out.logits.cpu()
    probs = torch.softmax(logits[0, -1, :], dim=0)

    def avg_first_token_prob(words: List[str]) -> float:
        ids = []
        for w in words:
            tok = tokenizer.encode(w, add_special_tokens=False)
            if tok:
                ids.append(tok[0])
        if not ids:
            return 0.0
        idx = torch.tensor(ids, dtype=torch.long)
        vals = probs.index_select(0, idx)
        return float(vals.mean().item())

    patched_clean = avg_first_token_prob(clean_keywords)
    patched_corr = avg_first_token_prob(corrupted_keywords)

    result = {
        'layer': layer,
        'baseline_clean_prob': float(baseline_clean),
        'baseline_corrupted_prob': float(baseline_corr),
        'patched_clean_prob': float(patched_clean),
        'patched_corrupted_prob': float(patched_corr),
        'delta_clean': float(patched_clean - baseline_clean),
        'delta_corrupted': float(patched_corr - baseline_corr),
        'baseline_corr_minus_clean': float(baseline_corr - baseline_clean),
        'patched_corr_minus_clean': float(patched_corr - patched_clean),
        'delta_corr_minus_clean': float((patched_corr - patched_clean) - (baseline_corr - baseline_clean)),
    }

    # Free memory
    del source_activation, logits, probs, traced_out, source_inputs, target_inputs
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return result


def scan_layer_nnsight(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str],
    corrupted_keywords: List[str],
    cache_file: Path,
    location: str = "post_attn",
    refresh_cache: bool = False
) -> List[Dict]:
    """Scan all heads in a layer with caching."""

    # Check cache (skip if refresh requested or cache is empty)
    if cache_file.exists() and not refresh_cache:
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            cached_results = cached.get('results', [])
            if cached_results:
                return cached_results
        except Exception:
            pass

    # For now, we scan entire layer as one unit
    # (NNsight head-level patching requires knowing model architecture)
    # This is a simplified version - scans layer-level only

    results = []

    # Layer-level patching first (head-level requires arch-specific hooks)
    result = patch_layer_nnsight(
        model, tokenizer, source_prompt, target_prompt, layer,
        clean_keywords, corrupted_keywords, location
    )
    result['head'] = 'all'  # Mark as layer-level
    results.append(result)

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
    parser.add_argument('--refresh-cache', action='store_true',
                       help='Clear cache before running')
    parser.add_argument('--location', type=str, default='post_attn',
                       choices=['pre_attn','post_attn','post_mlp'],
                       help='Residual location to patch')
    parser.add_argument('--layer', type=int, default=None,
                       help='If set, only patch this layer (0-index). Otherwise scan all layers')
    args = parser.parse_args()

    config = MATCHED_PAIRS[args.pair - 1]

    print("=" * 80)
    print(f"EXPERIMENT #10: NNsight Version (Pair {args.pair}/4)")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Pair: {config['baseline_id']} → {config['target_id']}")
    print(f"Location: {args.location}")

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Clear cache if requested
    if args.refresh_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print("\n✓ Cache cleared")

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
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    model = LanguageModel(args.model, device_map=DEVICE, dtype=dtype)
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

    # Corrupted keywords: default to using the corrupted twist phrase's first token
    corrupted_keywords = [config['corrupted_twist']]

    # Determine which layers to run
    if args.layer is not None:
        if args.layer < 0 or args.layer >= n_layers:
            print(f"Requested layer {args.layer} out of range [0,{n_layers-1}] — exiting")
            return
        layer_iter = [args.layer]
    else:
        layer_iter = list(range(n_layers))

    for layer in tqdm(layer_iter, desc="Layers"):
        cache_file = CACHE_DIR / f"pair{args.pair}_{args.location}_layer{layer:02d}.json"

        layer_results = scan_layer_nnsight(
            model, tokenizer, source_prompt, target_prompt,
            layer, config['clean_keywords'], corrupted_keywords,
            cache_file, location=args.location, refresh_cache=args.refresh_cache
        )

        all_results.extend(layer_results)

    # Sort and save
    # Rank by change in corrupted-minus-clean (override differential)
    all_results_sorted = sorted(all_results, key=lambda x: x.get('delta_corr_minus_clean', 0.0), reverse=True)

    output_file = OUTPUT_DIR / f"pair{args.pair}_nnsight_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'pair_config': config,
            'model': args.model,
            'total_layers': n_layers,
            'location': args.location,
            'results': all_results_sorted,
            'top_10': all_results_sorted[:10]
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Display
    print("\n" + "=" * 80)
    print("TOP 10 LAYERS")
    print("=" * 80)
    print(f"{'Layer':<8} {'bClean':<10} {'pClean':<10} {'Δclean':<9} {'b( corr-clean )':<18} {'p( corr-clean )':<18} {'Δdiff':<10}")
    print("-" * 80)

    if not all_results_sorted:
        print("(no results — try --refresh-cache or check keywords)")
    else:
        for result in all_results_sorted[:10]:
            print(
                f"{result['layer']:<8} "
                f"{result.get('baseline_clean_prob', 0.0):<10.6f} "
                f"{result.get('patched_clean_prob', 0.0):<10.6f} "
                f"{result.get('delta_clean', 0.0):+9.6f} "
                f"{result.get('baseline_corr_minus_clean', 0.0):<18.6f} "
                f"{result.get('patched_corr_minus_clean', 0.0):<18.6f} "
                f"{result.get('delta_corr_minus_clean', 0.0):+10.6f}"
            )


if __name__ == "__main__":
    main()
