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
from typing import Dict, List, Optional
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
MAX_INPUT_TOKENS: Optional[int] = None

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
    """Return traced handle for residual location: post_attn or post_mlp.

    For Qwen3 architecture:
    - post_attn: output of o_proj (after attention, before MLP)
    - post_mlp: output of entire transformer block
    """
    if location == "post_attn":
        return model.model.layers[layer].self_attn.o_proj.output
    if location == "post_mlp":
        return model.model.layers[layer].output
    raise ValueError("location must be: post_attn or post_mlp")


def load_prompts(baseline_id: str, target_id: str, think_mode: str) -> str:
    """Load the ablated prompt."""
    rollout_file = (
        ABLATION_DIR / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )
    with open(rollout_file, 'r') as f:
        data = json.load(f)
    return data['ablated_prompt']


def generate_and_count_keywords(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    clean_keywords: List[str],
    corrupted_keywords: List[str],
    max_new_tokens: int = 150
) -> Dict:
    """Generate story text and count both clean and corrupted keyword occurrences.

    Returns:
        {
            'clean_count': number of clean keyword occurrences,
            'corrupted_count': number of corrupted keyword occurrences,
            'total_tokens': number of tokens generated,
            'generated_text': the generated story (truncated to 200 chars)
        }
    """
    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(DEVICE)

    # Generate story (NNsight handles no_grad internally)
    output_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic for consistency
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode generated text (only new tokens)
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Count keywords (case-insensitive)
    generated_lower = generated_text.lower()

    clean_count = 0
    for keyword in clean_keywords:
        clean_count += generated_lower.count(keyword.lower())

    corrupted_count = 0
    for keyword in corrupted_keywords:
        corrupted_count += generated_lower.count(keyword.lower())

    # Free memory
    del inputs, output_ids, generated_ids
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return {
        'clean_count': clean_count,
        'corrupted_count': corrupted_count,
        'total_tokens': len(generated_text.split()),
        'generated_text': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text
    }


def patch_layer_nnsight(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    clean_keywords: List[str],
    corrupted_keywords: List[str],
    location: str = "post_attn",
    max_new_tokens: int = 150
) -> Dict:
    """Layer-level patch using NNsight - generates text and counts keywords."""

    # Baseline (no patching) - generate once and count both keyword sets
    baseline_result = generate_and_count_keywords(
        model, tokenizer, target_prompt, clean_keywords, corrupted_keywords, max_new_tokens
    )

    # Tokenize source prompt to capture activations
    source_inputs = tokenizer(
        source_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(DEVICE)

    # Capture source activation at the specified layer
    with torch.no_grad():
        with model.trace(source_inputs):
            src_act = get_resid_handle(model, layer, location).detach().save()
        source_activation = src_act

    # Tokenize target prompt for patched generation
    target_inputs = tokenizer(
        target_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(DEVICE)

    # Patched generation with manual loop - patch at EVERY generation step
    current_ids = target_inputs.input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Run forward pass with patch at layer L
            with model.trace(current_ids):
                tgt_handle = get_resid_handle(model, layer, location)

                # Patch: replace target activation with source activation
                # Only patch the prompt tokens (not generated tokens)
                prompt_len = target_inputs.input_ids.shape[1]
                min_len = min(source_activation.shape[1], prompt_len)
                tgt_handle[:, :min_len, :] = source_activation[:, :min_len, :]

                # Save logits for next token prediction
                next_logits = model.lm_head.output.save()

            # Get next token (greedy decoding)
            next_token_logits = next_logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if we hit EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    patched_output_ids = current_ids

    # Decode patched generation
    patched_generated_ids = patched_output_ids[0][target_inputs.input_ids.shape[1]:]
    patched_text = tokenizer.decode(patched_generated_ids, skip_special_tokens=True)

    # Count keywords in patched generation
    patched_lower = patched_text.lower()
    patched_clean_count = sum(patched_lower.count(kw.lower()) for kw in clean_keywords)
    patched_corrupted_count = sum(patched_lower.count(kw.lower()) for kw in corrupted_keywords)

    result = {
        'layer': layer,
        'baseline_clean_count': baseline_result['clean_count'],
        'baseline_corrupted_count': baseline_result['corrupted_count'],
        'baseline_total_tokens': baseline_result['total_tokens'],
        'baseline_text': baseline_result['generated_text'],
        'patched_clean_count': patched_clean_count,
        'patched_corrupted_count': patched_corrupted_count,
        'patched_total_tokens': len(patched_text.split()),
        'patched_text': patched_text[:200] + '...' if len(patched_text) > 200 else patched_text,
        'delta_clean': patched_clean_count - baseline_result['clean_count'],
        'delta_corrupted': patched_corrupted_count - baseline_result['corrupted_count'],
        'baseline_corr_minus_clean': baseline_result['corrupted_count'] - baseline_result['clean_count'],
        'patched_corr_minus_clean': patched_corrupted_count - patched_clean_count,
        'delta_corr_minus_clean': (patched_corrupted_count - patched_clean_count) -
                                   (baseline_result['corrupted_count'] - baseline_result['clean_count']),
    }

    # Free memory
    del source_activation, source_inputs, target_inputs, patched_output_ids, patched_generated_ids
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
    parser.add_argument('--max-input-tokens', type=int, default=None,
                       help='Truncate source/target prompts to this many tokens before tracing')
    parser.add_argument('--device', type=str, default=None, choices=['cpu','cuda'],
                       help='Force device placement; default auto-detect')
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

    # Configure globals
    global MAX_INPUT_TOKENS, DEVICE
    if args.device:
        DEVICE = args.device
    MAX_INPUT_TOKENS = args.max_input_tokens

    # Load model with NNsight
    print(f"\nLoading model with NNsight: {args.model}")
    print("This may take a few minutes...")

    # Load model with proper device handling for NNsight
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    try:
        # For NNsight, specify device via dispatch parameter
        if DEVICE == 'cuda':
            model = LanguageModel(
                args.model,
                torch_dtype=dtype,
                device_map='cuda',  # Use device_map for proper loading
                dispatch=True
            )
        else:
            model = LanguageModel(args.model, torch_dtype=dtype, dispatch=True)
    except RuntimeError as e:
        print(f"\n⚠ CUDA load failed ({e}); falling back to CPU")
        DEVICE = 'cpu'
        dtype = torch.float32
        model = LanguageModel(args.model, torch_dtype=dtype, dispatch=True)
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

        # Aggressive memory cleanup between layers (helps small GPUs)
        try:
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass

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
    print("TOP 10 LAYERS (ranked by effect on keyword counts)")
    print("=" * 80)
    print(f"{'Layer':<8} {'bClean':<8} {'pClean':<8} {'Δclean':<8} {'b(C-c)':<8} {'p(C-c)':<8} {'Δdiff':<8}")
    print("-" * 80)

    if not all_results_sorted:
        print("(no results — try --refresh-cache or check keywords)")
    else:
        for result in all_results_sorted[:10]:
            print(
                f"{result['layer']:<8} "
                f"{result.get('baseline_clean_count', 0):<8} "
                f"{result.get('patched_clean_count', 0):<8} "
                f"{result.get('delta_clean', 0):+8} "
                f"{result.get('baseline_corr_minus_clean', 0):<8} "
                f"{result.get('patched_corr_minus_clean', 0):<8} "
                f"{result.get('delta_corr_minus_clean', 0):+8}"
            )


if __name__ == "__main__":
    main()
