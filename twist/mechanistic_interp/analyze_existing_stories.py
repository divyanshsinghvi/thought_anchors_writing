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


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    # Split on period, exclamation, question mark followed by space or newline
    sentences = re.split(r'[.!?]+[\s\n]+', text)
    # Clean up and filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def semantic_similarity_check(
    story: str,
    twist_phrase: str,
    model: SentenceTransformer = None
) -> Dict:
    """
    Check semantic similarity between story sentences and twist phrase.

    Returns max similarity score and the most similar sentence.
    """
    if not SEMANTIC_AVAILABLE or model is None:
        return {
            'max_similarity': 0.0,
            'most_similar_sentence': '',
            'all_similarities': []
        }

    # Split story into sentences
    sentences = split_into_sentences(story)
    if not sentences:
        return {
            'max_similarity': 0.0,
            'most_similar_sentence': '',
            'all_similarities': []
        }

    # Encode twist and sentences
    twist_embedding = model.encode([twist_phrase], convert_to_numpy=True)[0]
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    # Compute cosine similarities
    similarities = []
    for sent_emb in sentence_embeddings:
        similarity = np.dot(twist_embedding, sent_emb) / (
            np.linalg.norm(twist_embedding) * np.linalg.norm(sent_emb)
        )
        similarities.append(float(similarity))

    # Find max
    max_idx = int(np.argmax(similarities))
    max_similarity = similarities[max_idx]
    most_similar_sentence = sentences[max_idx]

    return {
        'max_similarity': max_similarity,
        'most_similar_sentence': most_similar_sentence,
        'all_similarities': similarities,
        'num_sentences': len(sentences)
    }


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
    semantic_model: SentenceTransformer = None,
    use_semantic: bool = False
) -> Dict:
    """
    Analyze a baseline/ablation pair.

    Baseline: Outline has baseline_twist, Plan talks about baseline_twist
    Ablation: Outline has new_twist, Plan talks about baseline_twist (CONTAMINATED!)

    Check: Does ablation story use new_twist (Outline) or baseline_twist (Plan)?
    """
    # Extract stories
    baseline_story = baseline_data['response']['responses'][0]['content']
    ablation_story = ablation_data['rollout_data']['content']
    
    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']
    ablation_reasoning = ablation_data['rollout_data']['reasoning']

    # Extract twists
    baseline_twist = baseline_data['twist_phrase']
    new_twist = ablation_data['new_twist']

    # Check baseline story (should use baseline_twist)
    baseline_check = check_twist_in_story(baseline_story, baseline_twist)

    # Check ablation story for both twists
    ablation_baseline_check = check_twist_in_story(ablation_story, baseline_twist)  # Plan twist
    ablation_new_check = check_twist_in_story(ablation_story, new_twist)  # Outline twist

    # Semantic similarity checks (if enabled)
    baseline_semantic = {}
    ablation_baseline_semantic = {}
    ablation_new_semantic = {}

    if use_semantic and semantic_model is not None:
        baseline_semantic = semantic_similarity_check(baseline_story, baseline_twist, semantic_model)
        ablation_baseline_semantic = semantic_similarity_check(ablation_story, baseline_twist, semantic_model)
        ablation_new_semantic = semantic_similarity_check(ablation_story, new_twist, semantic_model)

    # Determine behavior (indicator-based)
    if ablation_new_check['twist_realized'] and not ablation_baseline_check['twist_realized']:
        behavior = "FOLLOWS_OUTLINE"
    elif ablation_baseline_check['twist_realized'] and not ablation_new_check['twist_realized']:
        behavior = "FOLLOWS_PLAN"
    elif ablation_new_check['twist_realized'] and ablation_baseline_check['twist_realized']:
        behavior = "USES_BOTH"
    else:
        behavior = "USES_NEITHER"

    # Determine behavior based on semantic similarity (if enabled)
    semantic_behavior = None
    if use_semantic and semantic_model is not None:
        plan_sim = ablation_baseline_semantic.get('max_similarity', 0.0)
        outline_sim = ablation_new_semantic.get('max_similarity', 0.0)

        # Use threshold and comparison
        THRESHOLD = 0.3  # Minimum similarity to consider
        if outline_sim > THRESHOLD and plan_sim <= THRESHOLD:
            semantic_behavior = "FOLLOWS_OUTLINE"
        elif plan_sim > THRESHOLD and outline_sim <= THRESHOLD:
            semantic_behavior = "FOLLOWS_PLAN"
        elif plan_sim > THRESHOLD and outline_sim > THRESHOLD:
            semantic_behavior = "USES_BOTH"
        else:
            semantic_behavior = "USES_NEITHER"

    result = {
        'baseline_id': baseline_data['prompt_id'],
        'ablation_id': ablation_data['twist_prompt_id'],
        'baseline_twist': baseline_twist,
        'new_twist': new_twist,
        'baseline_story': baseline_story,
        'ablation_story': ablation_story,
        'baseline_uses_correct_twist': baseline_check['twist_realized'],
        'baseline_indicators': baseline_check['indicators_found'],
        'ablation_uses_outline_twist': ablation_new_check['twist_realized'],
        'ablation_outline_indicators': ablation_new_check['indicators_found'],
        'ablation_uses_plan_twist': ablation_baseline_check['twist_realized'],
        'ablation_plan_indicators': ablation_baseline_check['indicators_found'],
        'model_behavior': behavior,
        'baseline_reasoning': baseline_reasoning,
        'ablation_reasoning': ablation_reasoning
    }

    # Add semantic similarity results if enabled
    if use_semantic and semantic_model is not None:
        result.update({
            'semantic_baseline_similarity': baseline_semantic.get('max_similarity', 0.0),
            'semantic_baseline_sentence': baseline_semantic.get('most_similar_sentence', ''),
            'semantic_plan_similarity': ablation_baseline_semantic.get('max_similarity', 0.0),
            'semantic_plan_sentence': ablation_baseline_semantic.get('most_similar_sentence', ''),
            'semantic_outline_similarity': ablation_new_semantic.get('max_similarity', 0.0),
            'semantic_outline_sentence': ablation_new_semantic.get('most_similar_sentence', ''),
            'semantic_behavior': semantic_behavior
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
        help='Use semantic similarity matching (requires sentence-transformers)'
    )
    parser.add_argument(
        '--semantic-model',
        type=str,
        default='all-mpnet-base-v2',
        help='Sentence transformer model name (default: all-mpnet-base-v2)'
    )

    args = parser.parse_args()

    # Load semantic model if requested
    semantic_model = None
    if args.use_semantic:
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

        print(f"  Baseline twist: '{result['baseline_twist']}'")
        print(f"  New twist (Outline): '{result['new_twist']}'")
        print(f"  Plan twist (contaminated): '{result['baseline_twist']}'")
        print(f"\n  Baseline story uses correct twist: {result['baseline_uses_correct_twist']}")
        print(f"    Indicators: {result['baseline_indicators']}")
        print(f"  Ablation story uses Outline twist: {result['ablation_uses_outline_twist']}")
        print(f"    Indicators: {result['ablation_outline_indicators']}")
        print(f"  Ablation story uses Plan twist: {result['ablation_uses_plan_twist']}")
        print(f"    Indicators: {result['ablation_plan_indicators']}")
        print(f"  → Model behavior (indicators): {result['model_behavior']}")

        # Semantic similarity results
        if args.use_semantic and 'semantic_behavior' in result:
            print(f"\n  Semantic Similarity:")
            print(f"    Outline twist similarity: {result['semantic_outline_similarity']:.3f}")
            print(f"      Most similar sentence: \"{result['semantic_outline_sentence'][:100]}...\"")
            print(f"    Plan twist similarity: {result['semantic_plan_similarity']:.3f}")
            print(f"      Most similar sentence: \"{result['semantic_plan_sentence'][:100]}...\"")
            print(f"  → Model behavior (semantic): {result['semantic_behavior']}")

        print(f"\n  Baseline story (first 200 chars):")
        print(f"    {result['baseline_story']}.")
        print(f"\n  Ablation story (first 200 chars):")
        print(f"    {result['ablation_story']}.")
        print(f"\n Baseline reasoning:")
        print(f"    {result['baseline_reasoning']}.")
        print(f"\n Ablation reasoning:")
        print(f"    {result['ablation_reasoning']}.")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    behaviors = [r['model_behavior'] for r in results]
    behavior_counts = {
        'FOLLOWS_OUTLINE': behaviors.count('FOLLOWS_OUTLINE'),
        'FOLLOWS_PLAN': behaviors.count('FOLLOWS_PLAN'),
        'USES_BOTH': behaviors.count('USES_BOTH'),
        'USES_NEITHER': behaviors.count('USES_NEITHER')
    }

    print(f"\nBehavior Distribution:")
    for behavior, count in behavior_counts.items():
        pct = (count / len(results)) * 100 if results else 0
        print(f"  {behavior}: {count}/{len(results)} ({pct:.1f}%)")

    # Interpretation
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    if behavior_counts['FOLLOWS_OUTLINE'] > behavior_counts['FOLLOWS_PLAN']:
        print("\n✓ Model primarily FOLLOWS OUTLINE (direct copy)")
        print("  → Constraint transfer is Outline → Story (bypasses Plan)")
    elif behavior_counts['FOLLOWS_PLAN'] > behavior_counts['FOLLOWS_OUTLINE']:
        print("\n✓ Model primarily FOLLOWS PLAN (CoT mediation)")
        print("  → Constraint transfer is Outline → Plan → Story")
    elif behavior_counts['USES_BOTH'] > len(results) / 2:
        print("\n⚠ Model uses BOTH twists (confused by conflict)")
        print("  → May indicate both Plan and Outline influence generation")
    else:
        print("\n⚠ Results MIXED or model uses NEITHER twist")
        print("  → Need more investigation or twists not literally present")

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
