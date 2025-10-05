#!/usr/bin/env python3
"""
Batch Semantic Analysis for All Ablation Experiments

This script processes all ablation experiments and computes semantic similarities
for each baseline × think_mode combination.

Output: JSON files with 8 semantic similarity metrics per ablation pair.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import re

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import OUTPUT_DIR_BASE_PATH

# Try to import embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class QwenEmbeddings:
    """Wrapper to use Qwen model for embeddings via mean pooling."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], convert_to_numpy: bool = True):
        """Encode texts to embeddings using mean pooling."""
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()

            return embeddings


def load_semantic_model(model_type: str = "qwen"):
    """Load semantic similarity model."""
    if model_type.lower() in ['qwen', 'qwen3-0.6b', 'qwen/qwen3-0.6b']:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install: pip install transformers torch")
        print(f"Loading Qwen model for embeddings...")
        return QwenEmbeddings("Qwen/Qwen3-0.6B")
    else:
        if not SEMANTIC_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install: pip install sentence-transformers")
        print(f"Loading sentence-transformers model: {model_type}...")
        return SentenceTransformer(model_type)


def compute_semantic_similarity(text1, text2, model) -> float:
    """Compute cosine similarity between two texts."""
    if model is None:
        return 0.0

    # Ensure both inputs are strings
    if not isinstance(text1, str):
        text1 = str(text1) if text1 is not None else ""
    if not isinstance(text2, str):
        text2 = str(text2) if text2 is not None else ""

    # Handle empty strings
    if not text1 or not text2:
        return 0.0

    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    emb1, emb2 = embeddings[0], embeddings[1]

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return float(similarity)


def load_baseline_story(baseline_id: str, stories_dir: Path) -> Dict:
    """Load baseline story."""
    baseline_file = stories_dir / f"{baseline_id}.json"
    with open(baseline_file, 'r') as f:
        return json.load(f)


def load_ablation_rollout(
    baseline_id: str,
    target_id: str,
    think_mode: str,
    ablation_dir: Path
) -> Dict:
    """Load ablation rollout."""
    rollout_file = (
        ablation_dir / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )

    if rollout_file.exists():
        with open(rollout_file, 'r') as f:
            return json.load(f)
    return None


def analyze_single_ablation(
    baseline_data: Dict,
    ablation_data: Dict,
    semantic_model
) -> Dict:
    """Compute 8 semantic similarities for one ablation pair."""

    # Extract data
    baseline_story = baseline_data['response']['responses'][0]['content']
    ablation_story = ablation_data['rollout_data']['content']

    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']

    # For ablation reasoning:
    # - no_more_thinking: skip computing ablation reasoning metrics
    # - allow_more_thinking: reasoning is new (model generated)
    ablation_reasoning = ablation_data['rollout_data'].get('reasoning')
    think_mode = ablation_data['ablation_metadata']['think_mode']
    analyze_reasoning = (think_mode == 'allow_more_thinking')

    clean_twist = baseline_data['twist_phrase']
    corrupted_twist = ablation_data['new_twist']

    # Compute semantic similarities
    baseline_content_vs_clean = compute_semantic_similarity(
        baseline_story, clean_twist, semantic_model
    )
    baseline_content_vs_corrupted = compute_semantic_similarity(
        baseline_story, corrupted_twist, semantic_model
    )
    ablation_content_vs_clean = compute_semantic_similarity(
        ablation_story, clean_twist, semantic_model
    )
    ablation_content_vs_corrupted = compute_semantic_similarity(
        ablation_story, corrupted_twist, semantic_model
    )
    baseline_reasoning_vs_clean = compute_semantic_similarity(
        baseline_reasoning, clean_twist, semantic_model
    )
    baseline_reasoning_vs_corrupted = compute_semantic_similarity(
        baseline_reasoning, corrupted_twist, semantic_model
    )

    result = {
        'baseline_id': baseline_data['prompt_id'],
        'target_id': ablation_data['twist_prompt_id'],
        'think_mode': think_mode,
        'clean_twist': clean_twist,
        'corrupted_twist': corrupted_twist,
        'baseline_content_vs_clean': baseline_content_vs_clean,
        'baseline_content_vs_corrupted': baseline_content_vs_corrupted,
        'ablation_content_vs_clean': ablation_content_vs_clean,
        'ablation_content_vs_corrupted': ablation_content_vs_corrupted,
        'baseline_reasoning_vs_clean': baseline_reasoning_vs_clean,
        'baseline_reasoning_vs_corrupted': baseline_reasoning_vs_corrupted,
    }

    # Determine content preference
    result['content_prefers'] = 'clean' if ablation_content_vs_clean > ablation_content_vs_corrupted else 'corrupted'

    # Reasoning metrics only for allow_more_thinking
    if analyze_reasoning and ablation_reasoning is not None:
        ablation_reasoning_vs_clean = compute_semantic_similarity(
            ablation_reasoning, clean_twist, semantic_model
        )
        ablation_reasoning_vs_corrupted = compute_semantic_similarity(
            ablation_reasoning, corrupted_twist, semantic_model
        )
        result['ablation_reasoning_vs_clean'] = ablation_reasoning_vs_clean
        result['ablation_reasoning_vs_corrupted'] = ablation_reasoning_vs_corrupted
        result['reasoning_prefers'] = 'clean' if ablation_reasoning_vs_clean > ablation_reasoning_vs_corrupted else 'corrupted'
    else:
        result['ablation_reasoning_vs_clean'] = None
        result['ablation_reasoning_vs_corrupted'] = None
        result['reasoning_prefers'] = None

    return result


def analyze_baseline_think_mode(
    baseline_id: str,
    think_mode: str,
    semantic_model,
    stories_dir: Path,
    ablation_dir: Path
) -> Dict:
    """Analyze all ablations for one baseline × think_mode combination."""

    print(f"\nAnalyzing {baseline_id} × {think_mode}")
    print("=" * 80)

    # Load baseline
    baseline_data = load_baseline_story(baseline_id, stories_dir)

    # Find all target ablations
    baseline_ablation_dir = ablation_dir / f"baseline_{baseline_id}"

    if not baseline_ablation_dir.exists():
        print(f"  ⚠ No ablations found for {baseline_id}")
        return None

    # Process each target
    ablations = []
    for target_dir in sorted(baseline_ablation_dir.glob("target_*")):
        target_id = target_dir.name.replace("target_", "")

        # Load ablation rollout
        ablation_data = load_ablation_rollout(
            baseline_id, target_id, think_mode, ablation_dir
        )

        if ablation_data is None:
            print(f"  ⚠ Skipping {target_id} - no rollout found")
            continue

        # Analyze
        result = analyze_single_ablation(
            baseline_data, ablation_data, semantic_model
        )
        ablations.append(result)

        reasoning_str = result['reasoning_prefers'] if result['reasoning_prefers'] is not None else 'skipped'
        print(f"  ✓ {target_id}: content={result['content_prefers']}, reasoning={reasoning_str}")

    # Compute summary statistics
    if ablations:
        content_prefers_clean = sum(1 for a in ablations if a['content_prefers'] == 'clean')

        # Common aggregates
        summary = {
            'total_ablations': len(ablations),
            'content_prefers_clean': content_prefers_clean,
            'content_prefers_corrupted': len(ablations) - content_prefers_clean,
            'avg_baseline_content_vs_clean': sum(a['baseline_content_vs_clean'] for a in ablations) / len(ablations),
            'avg_baseline_content_vs_corrupted': sum(a['baseline_content_vs_corrupted'] for a in ablations) / len(ablations),
            'avg_ablation_content_vs_clean': sum(a['ablation_content_vs_clean'] for a in ablations) / len(ablations),
            'avg_ablation_content_vs_corrupted': sum(a['ablation_content_vs_corrupted'] for a in ablations) / len(ablations),
            'avg_baseline_reasoning_vs_clean': sum(a['baseline_reasoning_vs_clean'] for a in ablations) / len(ablations),
            'avg_baseline_reasoning_vs_corrupted': sum(a['baseline_reasoning_vs_corrupted'] for a in ablations) / len(ablations),
        }

        if think_mode == 'allow_more_thinking':
            reasoning_prefers_clean = sum(1 for a in ablations if a['reasoning_prefers'] == 'clean')
            # Averages for ablation reasoning (valid only in allow_more_thinking)
            valid_clean = [a['ablation_reasoning_vs_clean'] for a in ablations if a['ablation_reasoning_vs_clean'] is not None]
            valid_corrupt = [a['ablation_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            avg_ablation_reasoning_vs_clean = sum(valid_clean) / len(valid_clean) if valid_clean else None
            avg_ablation_reasoning_vs_corrupted = sum(valid_corrupt) / len(valid_corrupt) if valid_corrupt else None

            # Correlation: increase in similarity (new twist vs reasoning) vs old reasoning similarity
            base_corr_list = [a['baseline_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            delta_corr_list = [a['ablation_reasoning_vs_corrupted'] - a['baseline_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            corr_delta_vs_old = None
            try:
                if len(base_corr_list) > 1:
                    import numpy as _np
                    if _np.std(base_corr_list) > 0 and _np.std(delta_corr_list) > 0:
                        corr_delta_vs_old = float(_np.corrcoef(base_corr_list, delta_corr_list)[0, 1])
            except Exception:
                corr_delta_vs_old = None

            summary.update({
                'reasoning_prefers_clean': reasoning_prefers_clean,
                'reasoning_prefers_corrupted': len(ablations) - reasoning_prefers_clean,
                'avg_ablation_reasoning_vs_clean': avg_ablation_reasoning_vs_clean,
                'avg_ablation_reasoning_vs_corrupted': avg_ablation_reasoning_vs_corrupted,
                'corr_delta_reasoning_vs_old_reasoning': corr_delta_vs_old,
            })
        else:
            # no_more_thinking: reasoning metrics are skipped
            summary.update({
                'reasoning_prefers_clean': None,
                'reasoning_prefers_corrupted': None,
                'avg_ablation_reasoning_vs_clean': None,
                'avg_ablation_reasoning_vs_corrupted': None,
                'corr_delta_reasoning_vs_old_reasoning': None,
            })

        print(f"\n  Summary:")
        print(f"    Content prefers clean: {content_prefers_clean}/{len(ablations)} ({100*content_prefers_clean/len(ablations):.1f}%)")
        if think_mode == 'allow_more_thinking':
            print(f"    Reasoning prefers clean: {summary['reasoning_prefers_clean']}/{len(ablations)} ({100*summary['reasoning_prefers_clean']/len(ablations):.1f}%)")
    else:
        summary = {}

    return {
        'baseline_id': baseline_id,
        'think_mode': think_mode,
        'ablations': ablations,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch semantic analysis for all ablation experiments'
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=['twist_001', 'twist_002', 'twist_003', 'twist_004', 'twist_005'],
        help='Baseline IDs to analyze (default: twist_001-005)'
    )
    parser.add_argument(
        '--think-modes',
        nargs='+',
        choices=['no_more_thinking', 'allow_more_thinking'],
        default=['no_more_thinking', 'allow_more_thinking'],
        help='Think modes to analyze'
    )
    parser.add_argument(
        '--semantic-model',
        type=str,
        default='qwen',
        help='Embedding model (default: qwen)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('twist/mechanistic_interp/outputs/batch_analysis'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Semantic Analysis - All Ablation Experiments")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Baselines: {args.baselines}")
    print(f"  Think modes: {args.think_modes}")
    print(f"  Semantic model: {args.semantic_model}")
    print(f"  Output dir: {args.output_dir}")

    # Load semantic model
    print(f"\nLoading semantic model...")
    semantic_model = load_semantic_model(args.semantic_model)
    print(f"✓ Model loaded")

    # Paths
    stories_dir = OUTPUT_DIR_BASE_PATH / "stories"
    ablation_dir = OUTPUT_DIR_BASE_PATH / "ablation_outputs"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each baseline × think_mode
    all_results = []

    for baseline_id in args.baselines:
        for think_mode in args.think_modes:
            result = analyze_baseline_think_mode(
                baseline_id,
                think_mode,
                semantic_model,
                stories_dir,
                ablation_dir
            )

            if result:
                all_results.append(result)

                # Save individual result
                output_file = args.output_dir / f"{baseline_id}_{think_mode}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n  ✓ Saved: {output_file}")

    # Save combined results
    combined_file = args.output_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Batch analysis complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Individual files: {len(all_results)}")
    print(f"  Combined file: {combined_file}")
    print(f"\nNext step:")
    print(f"  python twist/mechanistic_interp/cross_twist_analysis.py")


if __name__ == "__main__":
    main()
