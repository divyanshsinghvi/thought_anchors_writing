#!/usr/bin/env python3
"""
Setup and data preparation for mechanistic interpretability experiments.

This module:
1. Loads the Qwen3-14B model using TransformerLens
2. Prepares clean/corrupt prompt pairs from ablation data
3. Identifies token positions for Outline, Plan, and Story sections
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformer_lens import HookedTransformer

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import MODEL_NAME, OUTPUT_DIR_BASE_PATH

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = "Qwen/Qwen3-14B", use_cache: bool = True) -> HookedTransformer:
    """
    Load Qwen3-14B model using TransformerLens.

    Note: MODEL_NAME from config is "qwen/qwen3-14b" (OpenRouter format),
    but TransformerLens uses HuggingFace format "Qwen/Qwen3-14B"

    Args:
        model_name: HuggingFace model name
        use_cache: Whether to use local cache (default True)
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {DEVICE}")

    # Set cache directory
    cache_dir = OUTPUT_DIR_BASE_PATH / "transformerlens_cache" if use_cache else None

    model = HookedTransformer.from_pretrained(
        model_name,
        device=DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=str(cache_dir) if cache_dir else None
    )

    print(f"✓ Model loaded successfully")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Heads: {model.cfg.n_heads}")
    print(f"  d_model: {model.cfg.d_model}")

    return model


def load_ablation_data(
    baseline_id: str = "twist_001",
    ablation_dir: Path = OUTPUT_DIR_BASE_PATH / "ablation_outputs"
) -> Dict:
    """Load baseline and ablation data for experiments."""

    # Load baseline story
    baseline_file = OUTPUT_DIR_BASE_PATH / "stories" / f"{baseline_id}.json"
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    # Load ablation rollouts (different twists with same reasoning)
    ablation_data = {}

    for twist_dir in sorted(ablation_dir.glob("twist_*")):
        twist_id = twist_dir.name

        # Skip baseline itself
        if twist_id == baseline_id:
            continue

        # Load a sample rollout from no_more_thinking mode
        rollout_file = twist_dir / "no_more_thinking" / "rollout_000.json"
        if rollout_file.exists():
            with open(rollout_file, 'r') as f:
                ablation_data[twist_id] = json.load(f)

    return {
        'baseline': baseline_data,
        'ablations': ablation_data
    }


def create_clean_corrupt_pairs(data: Dict) -> List[Tuple[str, str, Dict]]:
    """
    Create clean/corrupt prompt pairs for logit difference experiments.

    Clean: Correct twist in Outline
    Corrupt: Different twist in Outline (using baseline reasoning)

    Returns list of (clean_prompt, corrupt_prompt, metadata) tuples
    """
    pairs = []

    baseline = data['baseline']
    baseline_twist = baseline['twist_phrase']
    baseline_protagonist = baseline['protagonist']

    for twist_id, ablation in data['ablations'].items():
        # Clean prompt: Original baseline with correct twist
        clean_prompt = baseline['combined_prompt']

        # Corrupt prompt: Ablation prompt with swapped twist (but same reasoning)
        corrupt_prompt = ablation['ablated_prompt']

        metadata = {
            'baseline_id': baseline['prompt_id'],
            'ablation_id': twist_id,
            'clean_twist': baseline_twist,
            'corrupt_twist': ablation['new_twist'],
            'protagonist': baseline_protagonist,
            'goal': baseline['goal']
        }

        pairs.append((clean_prompt, corrupt_prompt, metadata))

    return pairs


def identify_token_positions(
    model: HookedTransformer,
    prompt: str
) -> Dict[str, List[int]]:
    """
    Identify token positions for different sections of the prompt.

    Sections:
    - System: System prompt
    - Outline: "Outline:\nProtagonist:...\nGoal:...\nTwist:..."
    - Plan: "<think>...</think>" or "<think>..." (CoT reasoning)
    - Guidelines: "Guidelines:..."

    Returns dict mapping section names to token position lists
    """
    # Tokenize the full prompt
    tokens = model.to_tokens(prompt, prepend_bos=True)
    token_strs = model.to_str_tokens(prompt, prepend_bos=True)

    positions = {
        'system': [],
        'outline': [],
        'plan': [],
        'guidelines': [],
        'all': list(range(len(token_strs)))
    }

    # Find section boundaries using string matching
    full_text = prompt

    # System prompt ends at first "You will first think"
    outline_start_marker = "Outline:"
    plan_start_marker = "<think>"
    plan_end_marker = "</think>"
    guidelines_marker = "Guidelines:"

    current_section = 'system'
    in_plan = False

    for i, token_str in enumerate(token_strs):
        # Reconstruct approximate position in text
        # This is a heuristic - token positions don't map 1:1 to char positions

        if outline_start_marker in ''.join(token_strs[max(0, i-2):i+3]):
            current_section = 'outline'
        elif plan_start_marker in ''.join(token_strs[max(0, i-2):i+3]):
            current_section = 'plan'
            in_plan = True
        elif plan_end_marker in ''.join(token_strs[max(0, i-2):i+3]):
            in_plan = False
        elif guidelines_marker in ''.join(token_strs[max(0, i-2):i+3]):
            current_section = 'guidelines'

        if current_section in positions:
            positions[current_section].append(i)

    return positions


def extract_twist_tokens(
    model: HookedTransformer,
    prompt: str,
    twist_phrase: str
) -> List[int]:
    """
    Find token positions corresponding to the twist phrase in the Outline.

    Returns list of token indices
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    token_strs = model.to_str_tokens(prompt, prepend_bos=True)

    # Tokenize twist phrase to find it in the prompt
    twist_tokens = model.to_str_tokens(twist_phrase, prepend_bos=False)

    # Find where twist appears in the prompt tokens
    twist_positions = []
    for i in range(len(token_strs) - len(twist_tokens) + 1):
        if token_strs[i:i+len(twist_tokens)] == twist_tokens:
            twist_positions = list(range(i, i+len(twist_tokens)))
            break

    return twist_positions


def main():
    """Test setup and data preparation."""

    print("="*80)
    print("Mechanistic Interpretability Setup")
    print("="*80)

    # Load model
    model = load_model()

    # Load data
    print("\nLoading ablation data...")
    data = load_ablation_data()
    print(f"  Baseline: {data['baseline']['prompt_id']}")
    print(f"  Ablations: {list(data['ablations'].keys())}")

    # Create clean/corrupt pairs
    print("\nCreating clean/corrupt prompt pairs...")
    pairs = create_clean_corrupt_pairs(data)
    print(f"  Generated {len(pairs)} pairs")

    # Test token position identification
    print("\nTesting token position identification...")
    clean_prompt, corrupt_prompt, metadata = pairs[0]

    print(f"\nExample: {metadata['baseline_id']} → {metadata['ablation_id']}")
    print(f"  Clean twist: {metadata['clean_twist']}")
    print(f"  Corrupt twist: {metadata['corrupt_twist']}")

    clean_positions = identify_token_positions(model, clean_prompt)
    print(f"\nClean prompt token sections:")
    for section, positions in clean_positions.items():
        if positions:
            print(f"  {section}: {len(positions)} tokens (pos {positions[0]}-{positions[-1]})")

    # Extract twist token positions
    clean_twist_pos = extract_twist_tokens(model, clean_prompt, metadata['clean_twist'])
    corrupt_twist_pos = extract_twist_tokens(model, corrupt_prompt, metadata['corrupt_twist'])

    print(f"\nTwist token positions:")
    print(f"  Clean twist tokens: {clean_twist_pos}")
    print(f"  Corrupt twist tokens: {corrupt_twist_pos}")

    print("\n" + "="*80)
    print("Setup complete!")
    print("="*80)


if __name__ == "__main__":
    main()
