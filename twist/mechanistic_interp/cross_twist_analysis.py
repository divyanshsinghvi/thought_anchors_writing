#!/usr/bin/env python3
"""
Cross-Twist Meta-Analysis

Analyzes results across all baselines, think modes, and corrupted twists
to answer high-level questions about CoT mediation and constraint transfer.

Key Questions:
1. Does reasoning filter corrupted outlines? (CoT mediation)
2. Does think mode affect filtering?
3. Are some twists more "sticky" (harder to override)?
4. Are some baselines more resistant to corruption?
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all analysis results from batch analysis."""
    all_results = []

    combined_file = results_dir / "all_results.json"
    if combined_file.exists():
        with open(combined_file, 'r') as f:
            all_results = json.load(f)
    else:
        # Load individual files
        for result_file in sorted(results_dir.glob("twist_*.json")):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))

    return all_results


def analyze_cot_mediation(all_results: List[Dict]) -> Dict:
    """
    Question: Does reasoning filter corrupted outlines?

    Evidence for CoT mediation:
    - High % of ablation content prefers clean twist
    - Content similarity to clean > corrupted despite outline corruption
    """
    total_ablations = 0
    content_prefers_clean_count = 0
    reasoning_prefers_clean_count = 0

    ablation_content_clean_sims = []
    ablation_content_corrupt_sims = []

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        for ablation in result['ablations']:
            total_ablations += 1

            if ablation['content_prefers'] == 'clean':
                content_prefers_clean_count += 1
            if ablation['reasoning_prefers'] == 'clean':
                reasoning_prefers_clean_count += 1

            ablation_content_clean_sims.append(ablation['ablation_content_vs_clean'])
            ablation_content_corrupt_sims.append(ablation['ablation_content_vs_corrupted'])

    if total_ablations == 0:
        return {}

    content_clean_pct = 100 * content_prefers_clean_count / total_ablations
    reasoning_clean_pct = 100 * reasoning_prefers_clean_count / total_ablations

    avg_clean_sim = np.mean(ablation_content_clean_sims)
    avg_corrupt_sim = np.mean(ablation_content_corrupt_sims)

    # Interpretation
    if content_clean_pct > 70:
        interpretation = "STRONG CoT mediation - reasoning filters corrupted outline"
    elif content_clean_pct > 50:
        interpretation = "MODERATE CoT mediation - reasoning partially filters"
    elif content_clean_pct > 30:
        interpretation = "WEAK CoT mediation - mixed results"
    else:
        interpretation = "NO CoT mediation - outline leaks through"

    return {
        'total_ablations': total_ablations,
        'content_prefers_clean_count': content_prefers_clean_count,
        'content_prefers_clean_pct': content_clean_pct,
        'reasoning_prefers_clean_count': reasoning_prefers_clean_count,
        'reasoning_prefers_clean_pct': reasoning_clean_pct,
        'avg_ablation_content_vs_clean': avg_clean_sim,
        'avg_ablation_content_vs_corrupted': avg_corrupt_sim,
        'similarity_difference': avg_clean_sim - avg_corrupt_sim,
        'interpretation': interpretation
    }


def analyze_think_mode_effect(all_results: List[Dict]) -> Dict:
    """
    Question: Does think mode affect filtering?

    Compare no_more_thinking vs allow_more_thinking
    """
    by_think_mode = defaultdict(lambda: {
        'content_prefers_clean': 0,
        'total': 0,
        'content_clean_sims': [],
        'content_corrupt_sims': []
    })

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        think_mode = result['think_mode']

        for ablation in result['ablations']:
            by_think_mode[think_mode]['total'] += 1
            if ablation['content_prefers'] == 'clean':
                by_think_mode[think_mode]['content_prefers_clean'] += 1

            by_think_mode[think_mode]['content_clean_sims'].append(
                ablation['ablation_content_vs_clean']
            )
            by_think_mode[think_mode]['content_corrupt_sims'].append(
                ablation['ablation_content_vs_corrupted']
            )

    # Compute statistics
    stats = {}
    for mode, data in by_think_mode.items():
        if data['total'] > 0:
            stats[mode] = {
                'total': data['total'],
                'content_prefers_clean_count': data['content_prefers_clean'],
                'content_prefers_clean_pct': 100 * data['content_prefers_clean'] / data['total'],
                'avg_content_vs_clean': np.mean(data['content_clean_sims']),
                'avg_content_vs_corrupted': np.mean(data['content_corrupt_sims']),
            }

    # Compare modes
    if 'no_more_thinking' in stats and 'allow_more_thinking' in stats:
        no_more_pct = stats['no_more_thinking']['content_prefers_clean_pct']
        allow_more_pct = stats['allow_more_thinking']['content_prefers_clean_pct']
        difference = allow_more_pct - no_more_pct

        if abs(difference) < 5:
            interpretation = "No significant difference between think modes"
        elif difference > 0:
            interpretation = f"allow_more_thinking improves filtering by {difference:.1f}%"
        else:
            interpretation = f"no_more_thinking performs better by {-difference:.1f}%"

        stats['comparison'] = {
            'difference_pct': difference,
            'interpretation': interpretation
        }

    return stats


def analyze_twist_stickiness(all_results: List[Dict]) -> Dict:
    """
    Question: Are some corrupted twists more "sticky" (harder to override)?

    For each corrupted twist, measure how often it leaks into content.
    """
    by_corrupted_twist = defaultdict(lambda: {
        'total': 0,
        'content_prefers_corrupted': 0,
        'avg_corrupted_sim': []
    })

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        for ablation in result['ablations']:
            corrupted_twist = ablation['corrupted_twist']

            by_corrupted_twist[corrupted_twist]['total'] += 1
            if ablation['content_prefers'] == 'corrupted':
                by_corrupted_twist[corrupted_twist]['content_prefers_corrupted'] += 1

            by_corrupted_twist[corrupted_twist]['avg_corrupted_sim'].append(
                ablation['ablation_content_vs_corrupted']
            )

    # Compute stickiness scores
    stickiness = {}
    for twist, data in by_corrupted_twist.items():
        if data['total'] > 0:
            leak_pct = 100 * data['content_prefers_corrupted'] / data['total']
            avg_sim = np.mean(data['avg_corrupted_sim'])

            stickiness[twist] = {
                'total_occurrences': data['total'],
                'leak_count': data['content_prefers_corrupted'],
                'leak_pct': leak_pct,
                'avg_similarity': avg_sim,
                'stickiness_score': leak_pct * avg_sim  # Combined metric
            }

    # Rank by stickiness
    ranked = sorted(
        stickiness.items(),
        key=lambda x: x[1]['stickiness_score'],
        reverse=True
    )

    return {
        'by_twist': stickiness,
        'ranked': [(twist, data['stickiness_score']) for twist, data in ranked]
    }


def analyze_baseline_resistance(all_results: List[Dict]) -> Dict:
    """
    Question: Are some baselines more resistant to corruption?

    For each baseline, measure how well it filters corruption.
    """
    by_baseline = defaultdict(lambda: {
        'total': 0,
        'content_prefers_clean': 0,
        'clean_sims': [],
        'corrupt_sims': []
    })

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        baseline_id = result['baseline_id']

        for ablation in result['ablations']:
            by_baseline[baseline_id]['total'] += 1
            if ablation['content_prefers'] == 'clean':
                by_baseline[baseline_id]['content_prefers_clean'] += 1

            by_baseline[baseline_id]['clean_sims'].append(
                ablation['ablation_content_vs_clean']
            )
            by_baseline[baseline_id]['corrupt_sims'].append(
                ablation['ablation_content_vs_corrupted']
            )

    # Compute resistance scores
    resistance = {}
    for baseline, data in by_baseline.items():
        if data['total'] > 0:
            filter_pct = 100 * data['content_prefers_clean'] / data['total']
            avg_clean = np.mean(data['clean_sims'])
            avg_corrupt = np.mean(data['corrupt_sims'])
            sim_diff = avg_clean - avg_corrupt

            resistance[baseline] = {
                'total_ablations': data['total'],
                'filter_count': data['content_prefers_clean'],
                'filter_pct': filter_pct,
                'avg_clean_similarity': avg_clean,
                'avg_corrupt_similarity': avg_corrupt,
                'similarity_difference': sim_diff,
                'resistance_score': filter_pct * sim_diff  # Combined metric
            }

    # Rank by resistance
    ranked = sorted(
        resistance.items(),
        key=lambda x: x[1]['resistance_score'],
        reverse=True
    )

    return {
        'by_baseline': resistance,
        'ranked': [(baseline, data['resistance_score']) for baseline, data in ranked]
    }


def print_summary(analysis: Dict):
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("CROSS-TWIST META-ANALYSIS SUMMARY")
    print("=" * 80)

    # 1. CoT Mediation
    print("\n1. CoT MEDIATION ANALYSIS")
    print("-" * 80)
    cot = analysis['cot_mediation']
    if cot:
        print(f"Total ablations analyzed: {cot['total_ablations']}")
        print(f"\nContent Preference:")
        print(f"  Prefers clean: {cot['content_prefers_clean_count']} ({cot['content_prefers_clean_pct']:.1f}%)")
        print(f"  Prefers corrupted: {cot['total_ablations'] - cot['content_prefers_clean_count']} ({100 - cot['content_prefers_clean_pct']:.1f}%)")
        print(f"\nReasoning Preference:")
        print(f"  Focuses on clean: {cot['reasoning_prefers_clean_count']} ({cot['reasoning_prefers_clean_pct']:.1f}%)")
        print(f"\nSemantic Similarities:")
        print(f"  Avg ablation content vs clean:      {cot['avg_ablation_content_vs_clean']:.3f}")
        print(f"  Avg ablation content vs corrupted:  {cot['avg_ablation_content_vs_corrupted']:.3f}")
        print(f"  Difference:                          {cot['similarity_difference']:.3f}")
        print(f"\n→ {cot['interpretation']}")

    # 2. Think Mode Effect
    print("\n2. THINK MODE EFFECT")
    print("-" * 80)
    think_mode = analysis['think_mode_effect']
    for mode, stats in think_mode.items():
        if mode != 'comparison':
            print(f"\n{mode}:")
            print(f"  Total: {stats['total']}")
            print(f"  Content prefers clean: {stats['content_prefers_clean_count']} ({stats['content_prefers_clean_pct']:.1f}%)")
            print(f"  Avg clean sim: {stats['avg_content_vs_clean']:.3f}")
            print(f"  Avg corrupt sim: {stats['avg_content_vs_corrupted']:.3f}")

    if 'comparison' in think_mode:
        print(f"\n→ {think_mode['comparison']['interpretation']}")

    # 3. Twist Stickiness
    print("\n3. TWIST STICKINESS (Ranked)")
    print("-" * 80)
    stickiness = analysis['twist_stickiness']
    print(f"\nMost sticky (hardest to filter):")
    for i, (twist, score) in enumerate(stickiness['ranked'][:3], 1):
        data = stickiness['by_twist'][twist]
        print(f"  {i}. \"{twist}\"")
        print(f"     Leak rate: {data['leak_pct']:.1f}% ({data['leak_count']}/{data['total_occurrences']})")
        print(f"     Avg similarity: {data['avg_similarity']:.3f}")
        print(f"     Stickiness score: {score:.3f}")

    # 4. Baseline Resistance
    print("\n4. BASELINE RESISTANCE (Ranked)")
    print("-" * 80)
    resistance = analysis['baseline_resistance']
    print(f"\nMost resistant (best at filtering):")
    for i, (baseline, score) in enumerate(resistance['ranked'][:5], 1):
        data = resistance['by_baseline'][baseline]
        print(f"  {i}. {baseline}")
        print(f"     Filter rate: {data['filter_pct']:.1f}% ({data['filter_count']}/{data['total_ablations']})")
        print(f"     Similarity diff: {data['similarity_difference']:.3f}")
        print(f"     Resistance score: {score:.3f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-twist meta-analysis of ablation experiments'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('twist/mechanistic_interp/outputs/batch_analysis'),
        help='Directory containing batch analysis results'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('twist/mechanistic_interp/outputs/cross_twist_analysis.json'),
        help='Output file for meta-analysis results'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Cross-Twist Meta-Analysis")
    print("=" * 80)
    print(f"\nResults directory: {args.results_dir}")

    # Load all results
    print(f"\nLoading results...")
    all_results = load_all_results(args.results_dir)
    print(f"✓ Loaded {len(all_results)} result files")

    # Run analyses
    print(f"\nRunning meta-analyses...")

    analysis = {
        'cot_mediation': analyze_cot_mediation(all_results),
        'think_mode_effect': analyze_think_mode_effect(all_results),
        'twist_stickiness': analyze_twist_stickiness(all_results),
        'baseline_resistance': analyze_baseline_resistance(all_results),
    }

    # Print summary
    print_summary(analysis)

    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Meta-analysis results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
