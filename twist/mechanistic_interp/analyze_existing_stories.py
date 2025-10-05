#!/usr/bin/env python3
"""
Analyze Existing Generated Stories for Twist Realization

The ablation data already has generated stories in rollout_data['content'].
This script analyzes those stories to see which twist the model used.

Key Question: When Outline says X but Plan talks about Y, which does the model follow?
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

# Try to import sentence transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠ Warning: sentence-transformers not installed. Semantic matching unavailable.")

# Try to import transformers for Qwen embeddings
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ Warning: transformers not installed. Qwen embeddings unavailable.")


class QwenEmbeddings:
    """Wrapper to use Qwen model for embeddings via mean pooling."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
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
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**inputs)

            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()

            return embeddings


def load_ablation_data(
    baseline_id: str = "twist_001",
    ablation_dir: Path = OUTPUT_DIR_BASE_PATH / "ablation_outputs",
    think_mode: str = "no_more_thinking",
    max_rollouts_per_twist: int = None
) -> dict:
    """
    Load baseline and ablation data for experiments.

    Args:
        baseline_id: ID of baseline story (e.g., "twist_001")
        ablation_dir: Directory containing ablation outputs
        think_mode: Which think mode to analyze ("no_more_thinking" or "allow_more_thinking")
        max_rollouts_per_twist: Max rollouts to load per twist (None = all)
    """
    baseline_file = OUTPUT_DIR_BASE_PATH / "stories" / f"{baseline_id}.json"
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    ablation_data = {}

    # Support both old (twist_NNN/) and new (baseline_XXX/target_YYY/) structures
    # Try new structure first
    baseline_dirs = list(ablation_dir.glob(f"baseline_{baseline_id}"))

    if baseline_dirs:
        # New structure: baseline_XXX/target_YYY/think_mode/
        for baseline_dir in baseline_dirs:
            for target_dir in sorted(baseline_dir.glob("target_*")):
                target_id = target_dir.name.replace("target_", "")

                rollout_dir = target_dir / think_mode
                if not rollout_dir.exists():
                    continue

                # Load all rollouts (or up to max)
                rollout_files = sorted(rollout_dir.glob("rollout_*.json"))
                if max_rollouts_per_twist:
                    rollout_files = rollout_files[:max_rollouts_per_twist]

                for rollout_file in rollout_files:
                    with open(rollout_file, 'r') as f:
                        rollout_data = json.load(f)
                        rollout_key = f"{target_id}_{rollout_file.stem}"
                        ablation_data[rollout_key] = rollout_data
    else:
        # Old structure: twist_NNN/think_mode/
        for twist_dir in sorted(ablation_dir.glob("twist_*")):
            twist_id = twist_dir.name
            if twist_id == baseline_id:
                continue

            rollout_dir = twist_dir / think_mode
            if not rollout_dir.exists():
                continue

            # Load all rollouts (or up to max)
            rollout_files = sorted(rollout_dir.glob("rollout_*.json"))
            if max_rollouts_per_twist:
                rollout_files = rollout_files[:max_rollouts_per_twist]

            for rollout_file in rollout_files:
                with open(rollout_file, 'r') as f:
                    rollout_data = json.load(f)
                    rollout_key = f"{twist_id}_{rollout_file.stem}"
                    ablation_data[rollout_key] = rollout_data

    return {'baseline': baseline_data, 'ablations': ablation_data}


def compute_semantic_similarity(
    text1: str,
    text2: str,
    model = None  # Can be SentenceTransformer or QwenEmbeddings
) -> float:
    """
    Compute semantic similarity between two texts using full paragraph embeddings.

    Args:
        text1: First text (e.g., story content)
        text2: Second text (e.g., twist phrase)
        model: Embedding model (SentenceTransformer or QwenEmbeddings)

    Returns:
        Cosine similarity score (0-1)
    """
    if model is None:
        return 0.0

    # Encode both texts
    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    emb1, emb2 = embeddings[0], embeddings[1]

    # Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return float(similarity)


def check_twist_in_story(story: str, twist_phrase: str) -> Dict:
    """
    Check if twist appears in story (literal or narrative indicators).

    Twist indicators:
    - "it was only a dream": woke, waking, dream, dreaming, nightmare
    - "they were dead all along": dead, death, ghost, spirit, afterlife, corpse
    - "villain was their future self": future, time, older, paradox
    - "world is a simulation": simulation, program, code, matrix, virtual, glitch
    - "last human alive": alone, last, empty, desolate, abandoned
    """
    story_lower = story.lower()
    twist_lower = twist_phrase.lower()

    # Literal match
    twist_realized_literal = twist_lower in story_lower

    # Narrative indicators for each twist type
    indicators_found = []

    if "dream" in twist_lower:
        dream_indicators = ['woke', 'waking', 'dream', 'dreaming', 'nightmare', 'asleep', 'sleeping']
        indicators_found = [ind for ind in dream_indicators if ind in story_lower]

    elif "dead" in twist_lower:
        death_indicators = ['dead', 'death', 'ghost', 'spirit', 'afterlife', 'corpse', 'died', 'lifeless']
        indicators_found = [ind for ind in death_indicators if ind in story_lower]

    elif "future self" in twist_lower or "villain" in twist_lower:
        time_indicators = ['future', 'time', 'older', 'paradox', 'past', 'years']
        indicators_found = [ind for ind in time_indicators if ind in story_lower]

    elif "simulation" in twist_lower:
        sim_indicators = ['simulation', 'program', 'code', 'matrix', 'virtual', 'glitch', 'pixel']
        indicators_found = [ind for ind in sim_indicators if ind in story_lower]

    elif "last human" in twist_lower or "alone" in twist_lower:
        alone_indicators = ['alone', 'last', 'empty', 'desolate', 'abandoned', 'solitary', 'nobody']
        indicators_found = [ind for ind in alone_indicators if ind in story_lower]

    # Consider twist realized if literal OR strong narrative indicators
    twist_realized = twist_realized_literal or len(indicators_found) >= 1

    return {
        'twist_realized_literal': twist_realized_literal,
        'twist_realized': twist_realized,
        'indicators_found': indicators_found,
        'num_indicators': len(indicators_found)
    }


def analyze_story_pair(
    baseline_data: dict,
    ablation_data: dict,
    semantic_model = None,
    use_semantic: bool = False
) -> Dict:
    """
    Analyze a baseline/ablation pair with semantic similarity.

    Key metrics (4 semantic similarities):
    1. baseline_content vs clean_twist (should be high)
    2. baseline_content vs corrupted_twist (should be low)
    3. ablation_content vs clean_twist (tests if reasoning filters corruption)
    4. ablation_content vs corrupted_twist (tests if outline corrupts output)

    Additional:
    5. baseline_reasoning vs clean_twist
    6. baseline_reasoning vs corrupted_twist
    7. ablation_reasoning vs clean_twist
    8. ablation_reasoning vs corrupted_twist
    """
    # Extract data
    baseline_story = baseline_data['response']['responses'][0]['content']
    ablation_story = ablation_data['rollout_data']['content']

    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']
    ablation_reasoning = ablation_data['rollout_data']['reasoning']

    # Twists
    clean_twist = baseline_data['twist_phrase']  # The correct twist
    corrupted_twist = ablation_data['new_twist']  # The swapped twist in ablation

    # Check baseline story (indicator-based, keep for reference)
    baseline_check = check_twist_in_story(baseline_story, clean_twist)
    ablation_clean_check = check_twist_in_story(ablation_story, clean_twist)
    ablation_corrupted_check = check_twist_in_story(ablation_story, corrupted_twist)

    result = {
        'baseline_id': baseline_data['prompt_id'],
        'ablation_id': ablation_data['twist_prompt_id'],
        'clean_twist': clean_twist,
        'corrupted_twist': corrupted_twist,
        'baseline_story': baseline_story,
        'ablation_story': ablation_story,
        'baseline_reasoning': baseline_reasoning,
        'ablation_reasoning': ablation_reasoning,

        # Indicator-based (legacy)
        'baseline_has_clean_twist_indicator': baseline_check['twist_realized'],
        'ablation_has_clean_twist_indicator': ablation_clean_check['twist_realized'],
        'ablation_has_corrupted_twist_indicator': ablation_corrupted_check['twist_realized'],
    }

    # Semantic similarity analysis (if enabled)
    if use_semantic and semantic_model is not None:
        # Core 4 metrics: Content similarities
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

        # Additional: Reasoning similarities
        baseline_reasoning_vs_clean = compute_semantic_similarity(
            baseline_reasoning, clean_twist, semantic_model
        )
        baseline_reasoning_vs_corrupted = compute_semantic_similarity(
            baseline_reasoning, corrupted_twist, semantic_model
        )
        ablation_reasoning_vs_clean = compute_semantic_similarity(
            ablation_reasoning, clean_twist, semantic_model
        )
        ablation_reasoning_vs_corrupted = compute_semantic_similarity(
            ablation_reasoning, corrupted_twist, semantic_model
        )

        result.update({
            # Core 4 metrics
            'baseline_content_vs_clean': baseline_content_vs_clean,
            'baseline_content_vs_corrupted': baseline_content_vs_corrupted,
            'ablation_content_vs_clean': ablation_content_vs_clean,
            'ablation_content_vs_corrupted': ablation_content_vs_corrupted,

            # Reasoning analysis
            'baseline_reasoning_vs_clean': baseline_reasoning_vs_clean,
            'baseline_reasoning_vs_corrupted': baseline_reasoning_vs_corrupted,
            'ablation_reasoning_vs_clean': ablation_reasoning_vs_clean,
            'ablation_reasoning_vs_corrupted': ablation_reasoning_vs_corrupted,
        })

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Analyze existing generated stories for twist realization'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='twist_001',
        help='Baseline story ID (default: twist_001)'
    )
    parser.add_argument(
        '--think-mode',
        type=str,
        choices=['no_more_thinking', 'allow_more_thinking'],
        default='no_more_thinking',
        help='Which think mode to analyze (default: no_more_thinking)'
    )
    parser.add_argument(
        '--max-rollouts',
        type=int,
        default=None,
        help='Max rollouts to analyze per twist (default: all)'
    )
    parser.add_argument(
        '--use-semantic',
        action='store_true',
        help='Use semantic similarity matching (requires sentence-transformers or transformers)'
    )
    parser.add_argument(
        '--semantic-model',
        type=str,
        default='all-mpnet-base-v2',
        help='Model name for embeddings (default: all-mpnet-base-v2). Use "qwen" for Qwen/Qwen2.5-0.5B'
    )

    args = parser.parse_args()

    # Load semantic model if requested
    semantic_model = None
    if args.use_semantic:
        if args.semantic_model.lower() in ['qwen', 'qwen3-0.6b', 'qwen/qwen3-0.6b']:
            # Use Qwen for embeddings
            if not TRANSFORMERS_AVAILABLE:
                print("\n⚠ Error: transformers not installed. Install with:")
                print("  pip install transformers torch")
                return
            print(f"\nLoading Qwen model for embeddings: Qwen/Qwen3-0.6B...")
            semantic_model = QwenEmbeddings("Qwen/Qwen3-0.6B")
            print(f"✓ Qwen model loaded")
        else:
            # Use sentence-transformers
            if not SEMANTIC_AVAILABLE:
                print("\n⚠ Error: sentence-transformers not installed. Install with:")
                print("  pip install sentence-transformers")
                return
            print(f"\nLoading semantic model: {args.semantic_model}...")
            semantic_model = SentenceTransformer(args.semantic_model)
            print(f"✓ Model loaded")

    print("="*80)
    print("Analyzing Existing Generated Stories")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Baseline: {args.baseline}")
    print(f"  Think mode: {args.think_mode}")
    print(f"  Max rollouts per twist: {args.max_rollouts or 'all'}")
    print(f"  Semantic matching: {args.use_semantic}")

    # Load data
    print("\nLoading ablation data...")
    data = load_ablation_data(
        baseline_id=args.baseline,
        think_mode=args.think_mode,
        max_rollouts_per_twist=args.max_rollouts
    )
    print(f"  Baseline: {data['baseline']['prompt_id']}")
    print(f"  Total ablation rollouts loaded: {len(data['ablations'])}")

    if not data['ablations']:
        print("\n⚠ No ablation data found. Run twist_ablation.py first.")
        return

    # Analyze each pair
    print("\n" + "="*80)
    print("Twist Realization Analysis")
    print("="*80)

    results = []
    for twist_id, ablation in data['ablations'].items():
        print(f"\nAnalyzing: {data['baseline']['prompt_id']} → {twist_id}")

        result = analyze_story_pair(
            data['baseline'],
            ablation,
            semantic_model=semantic_model,
            use_semantic=args.use_semantic
        )
        results.append(result)

        print(f"  Clean twist: '{result['clean_twist']}'")
        print(f"  Corrupted twist: '{result['corrupted_twist']}'")

        # Semantic similarity results
        if args.use_semantic and 'baseline_content_vs_clean' in result:
            print(f"\n  CONTENT Semantic Similarities:")
            print(f"    Baseline content vs clean twist:      {result['baseline_content_vs_clean']:.3f}")
            print(f"    Baseline content vs corrupted twist:  {result['baseline_content_vs_corrupted']:.3f}")
            print(f"    Ablation content vs clean twist:      {result['ablation_content_vs_clean']:.3f}")
            print(f"    Ablation content vs corrupted twist:  {result['ablation_content_vs_corrupted']:.3f}")

            print(f"\n  REASONING Semantic Similarities:")
            print(f"    Baseline reasoning vs clean twist:      {result['baseline_reasoning_vs_clean']:.3f}")
            print(f"    Baseline reasoning vs corrupted twist:  {result['baseline_reasoning_vs_corrupted']:.3f}")
            print(f"    Ablation reasoning vs clean twist:      {result['ablation_reasoning_vs_clean']:.3f}")
            print(f"    Ablation reasoning vs corrupted twist:  {result['ablation_reasoning_vs_corrupted']:.3f}")

            # Analysis
            content_prefers_clean = result['ablation_content_vs_clean'] > result['ablation_content_vs_corrupted']
            reasoning_prefers_clean = result['ablation_reasoning_vs_clean'] > result['ablation_reasoning_vs_corrupted']

            print(f"\n  Analysis:")
            print(f"    Ablation content prefers: {'CLEAN' if content_prefers_clean else 'CORRUPTED'}")
            print(f"    Ablation reasoning prefers: {'CLEAN' if reasoning_prefers_clean else 'CORRUPTED'}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if args.use_semantic:
        # Compute aggregate statistics
        baseline_clean_sims = [r.get('baseline_content_vs_clean', 0) for r in results if 'baseline_content_vs_clean' in r]
        baseline_corrupt_sims = [r.get('baseline_content_vs_corrupted', 0) for r in results if 'baseline_content_vs_corrupted' in r]
        ablation_clean_sims = [r.get('ablation_content_vs_clean', 0) for r in results if 'ablation_content_vs_clean' in r]
        ablation_corrupt_sims = [r.get('ablation_content_vs_corrupted', 0) for r in results if 'ablation_content_vs_corrupted' in r]

        baseline_reasoning_clean = [r.get('baseline_reasoning_vs_clean', 0) for r in results if 'baseline_reasoning_vs_clean' in r]
        baseline_reasoning_corrupt = [r.get('baseline_reasoning_vs_corrupted', 0) for r in results if 'baseline_reasoning_vs_corrupted' in r]
        ablation_reasoning_clean = [r.get('ablation_reasoning_vs_clean', 0) for r in results if 'ablation_reasoning_vs_clean' in r]
        ablation_reasoning_corrupt = [r.get('ablation_reasoning_vs_corrupted', 0) for r in results if 'ablation_reasoning_vs_corrupted' in r]

        if baseline_clean_sims:
            print(f"\nCONTENT Semantic Similarity Averages:")
            print(f"  Baseline content vs clean:      {sum(baseline_clean_sims)/len(baseline_clean_sims):.3f}")
            print(f"  Baseline content vs corrupted:  {sum(baseline_corrupt_sims)/len(baseline_corrupt_sims):.3f}")
            print(f"  Ablation content vs clean:      {sum(ablation_clean_sims)/len(ablation_clean_sims):.3f}")
            print(f"  Ablation content vs corrupted:  {sum(ablation_corrupt_sims)/len(ablation_corrupt_sims):.3f}")

            print(f"\nREASONING Semantic Similarity Averages:")
            print(f"  Baseline reasoning vs clean:      {sum(baseline_reasoning_clean)/len(baseline_reasoning_clean):.3f}")
            print(f"  Baseline reasoning vs corrupted:  {sum(baseline_reasoning_corrupt)/len(baseline_reasoning_corrupt):.3f}")
            print(f"  Ablation reasoning vs clean:      {sum(ablation_reasoning_clean)/len(ablation_reasoning_clean):.3f}")
            print(f"  Ablation reasoning vs corrupted:  {sum(ablation_reasoning_corrupt)/len(ablation_reasoning_corrupt):.3f}")

            # Count preferences
            content_prefers_clean = sum(1 for r in results if r.get('ablation_content_vs_clean', 0) > r.get('ablation_content_vs_corrupted', 0))
            content_prefers_corrupt = sum(1 for r in results if r.get('ablation_content_vs_clean', 0) < r.get('ablation_content_vs_corrupted', 0))
            reasoning_prefers_clean = sum(1 for r in results if r.get('ablation_reasoning_vs_clean', 0) > r.get('ablation_reasoning_vs_corrupted', 0))
            reasoning_prefers_corrupt = sum(1 for r in results if r.get('ablation_reasoning_vs_clean', 0) < r.get('ablation_reasoning_vs_corrupted', 0))

            print(f"\nPreference Counts:")
            print(f"  Ablation CONTENT prefers clean:      {content_prefers_clean}/{len(results)} ({100*content_prefers_clean/len(results):.1f}%)")
            print(f"  Ablation CONTENT prefers corrupted:  {content_prefers_corrupt}/{len(results)} ({100*content_prefers_corrupt/len(results):.1f}%)")
            print(f"  Ablation REASONING prefers clean:      {reasoning_prefers_clean}/{len(results)} ({100*reasoning_prefers_clean/len(results):.1f}%)")
            print(f"  Ablation REASONING prefers corrupted:  {reasoning_prefers_corrupt}/{len(results)} ({100*reasoning_prefers_corrupt/len(results):.1f}%)")

    # Interpretation
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    if args.use_semantic and baseline_clean_sims:
        avg_ablation_clean = sum(ablation_clean_sims) / len(ablation_clean_sims)
        avg_ablation_corrupt = sum(ablation_corrupt_sims) / len(ablation_corrupt_sims)
        avg_reasoning_clean = sum(ablation_reasoning_clean) / len(ablation_reasoning_clean)
        avg_reasoning_corrupt = sum(ablation_reasoning_corrupt) / len(ablation_reasoning_corrupt)

        print(f"\nKey findings:")

        if content_prefers_clean > content_prefers_corrupt * 1.5:
            print(f"  ✓ Ablation content prefers CLEAN twist ({100*content_prefers_clean/len(results):.0f}%)")
            print(f"    → Reasoning successfully filters corrupted outline")
            print(f"    → Evidence for CoT mediation: Outline → Reasoning → Content")
        elif content_prefers_corrupt > content_prefers_clean * 1.5:
            print(f"  ✓ Ablation content prefers CORRUPTED twist ({100*content_prefers_corrupt/len(results):.0f}%)")
            print(f"    → Corrupted outline leaks directly into content")
            print(f"    → Evidence for direct path: Outline → Content (bypasses reasoning)")
        else:
            print(f"  ⚠ Mixed results: Content shows no clear preference")

        if reasoning_prefers_clean > reasoning_prefers_corrupt * 1.5:
            print(f"\n  ✓ Ablation reasoning focuses on CLEAN twist ({100*reasoning_prefers_clean/len(results):.0f}%)")
            print(f"    → Reasoning ignores corrupted outline, uses original plan")
        elif reasoning_prefers_corrupt > reasoning_prefers_clean * 1.5:
            print(f"\n  ✓ Ablation reasoning focuses on CORRUPTED twist ({100*reasoning_prefers_corrupt/len(results):.0f}%)")
            print(f"    → Reasoning adapts to corrupted outline")
    else:
        print(f"\n⚠ No semantic analysis performed. Use --use-semantic for detailed analysis.")

    # Save results
    output_file = Path("twist/mechanistic_interp/outputs/story_analysis_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
