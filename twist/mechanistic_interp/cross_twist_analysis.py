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
5. How do LLM twist detections compare with semantic similarity shifts?
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

PRESENCE_FIELDS = {
    'baseline_presence': 'Baseline story',
    'ablation_presence': 'Ablation story',
    'baseline_reasoning_presence': 'Baseline reasoning',
    'ablation_reasoning_presence': 'Ablation reasoning'
}

PRESENCE_THINK_MODES = {
    'baseline_presence': 'any',
    'ablation_presence': 'any',
    'baseline_reasoning_presence': 'any',
    'ablation_reasoning_presence': 'allow_more_thinking',
}


def _init_presence_counter():
    return {
        'clean': {
            'requested': 0,
            'parsed': 0,
            'present_total': 0,
            'present_true': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'present_pct': None,
            'parsed_pct': None,
            'confidence_avg': None,
        },
        'corrupted': {
            'requested': 0,
            'parsed': 0,
            'present_total': 0,
            'present_true': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
            'present_pct': None,
            'parsed_pct': None,
            'confidence_avg': None,
        }
    }


def _finalize_presence_counter(counter: Dict):
    for key in ['clean', 'corrupted']:
        data = counter[key]
        if data['present_total'] > 0:
            data['present_pct'] = 100 * data['present_true'] / data['present_total']
        if data['requested'] > 0:
            data['parsed_pct'] = 100 * data['parsed'] / data['requested']
        if data['confidence_count'] > 0:
            data['confidence_avg'] = data['confidence_sum'] / data['confidence_count']

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
    reasoning_total = 0

    ablation_content_clean_sims = []
    ablation_content_corrupt_sims = []

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        for ablation in result['ablations']:
            total_ablations += 1

            if ablation['content_prefers'] == 'clean':
                content_prefers_clean_count += 1

            reasoning_pref = ablation.get('reasoning_prefers')
            if reasoning_pref is not None:
                reasoning_total += 1
                if reasoning_pref == 'clean':
                    reasoning_prefers_clean_count += 1

            ablation_content_clean_sims.append(ablation['ablation_content_vs_clean'])
            ablation_content_corrupt_sims.append(ablation['ablation_content_vs_corrupted'])

    if total_ablations == 0:
        return {}

    content_clean_pct = 100 * content_prefers_clean_count / total_ablations
    reasoning_clean_pct = (
        100 * reasoning_prefers_clean_count / reasoning_total
        if reasoning_total > 0 else None
    )

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
        'reasoning_total_ablations': reasoning_total,
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


def analyze_reasoning_shift(all_results: List[Dict]) -> Dict:
    """Aggregate reasoning shift statistics for allow_more_thinking runs."""
    baselines = []
    deltas = []
    content_prefers_corrupted = 0
    total_with_reasoning = 0

    for result in all_results:
        if not result or result.get('think_mode') != 'allow_more_thinking':
            continue

        for ablation in result.get('ablations', []):
            baseline_corr = ablation.get('baseline_reasoning_vs_corrupted')
            ablation_corr = ablation.get('ablation_reasoning_vs_corrupted')
            if baseline_corr is None or ablation_corr is None:
                continue

            baselines.append(baseline_corr)
            deltas.append(ablation_corr - baseline_corr)
            total_with_reasoning += 1
            if ablation.get('content_prefers') == 'corrupted':
                content_prefers_corrupted += 1

    if total_with_reasoning < 2:
        return {
            'total_points': total_with_reasoning,
            'correlation': None,
            'mean_baseline_similarity': None,
            'mean_shift': None,
            'content_prefers_corrupted_pct': None,
        }

    baselines_arr = np.array(baselines)
    deltas_arr = np.array(deltas)

    valid = np.std(baselines_arr) > 0 and np.std(deltas_arr) > 0
    correlation = float(np.corrcoef(baselines_arr, deltas_arr)[0, 1]) if valid else None

    return {
        'total_points': total_with_reasoning,
        'correlation': correlation,
        'mean_baseline_similarity': float(np.mean(baselines_arr)),
        'mean_shift': float(np.mean(deltas_arr)),
        'content_prefers_corrupted_pct': (
            100 * content_prefers_corrupted / total_with_reasoning
            if total_with_reasoning else None
        ),
    }


def analyze_presence_detection(all_results: List[Dict]) -> Dict:
    """Aggregate LLM twist presence detection statistics."""

    presence_summary = {field: _init_presence_counter() for field in PRESENCE_FIELDS}
    presence_by_mode = {
        'no_more_thinking': {field: _init_presence_counter() for field in PRESENCE_FIELDS},
        'allow_more_thinking': {field: _init_presence_counter() for field in PRESENCE_FIELDS},
    }
    presence_models = set()
    thresholds = set()
    batch_sizes = set()
    max_new_tokens = set()
    total_parse_warnings = 0

    any_requested = False

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        models_info = result.get('models', {}) or {}
        result_presence_model = models_info.get('presence')
        result_threshold = None
        result_batch = None
        result_max_new = None

        # Summary-level metadata
        summary = result.get('summary') or {}
        if isinstance(summary, dict):
            result_threshold = summary.get('presence_threshold')
            result_batch = summary.get('presence_batch_size')
            result_max_new = summary.get('presence_max_new_tokens')
            total_parse_warnings += summary.get('presence_parse_warnings', 0) or 0

        for ablation in result['ablations']:
            presence_model = ablation.get('presence_model') or result_presence_model
            if presence_model:
                presence_models.add(presence_model)

            fields_requested = set(ablation.get('presence_fields_requested') or [])
            think_mode = result.get('think_mode')
            for field, label in PRESENCE_FIELDS.items():
                counter = presence_summary[field]
                mode_counter = presence_by_mode.get(think_mode, {}).get(field)

                if field in fields_requested:
                    counter['clean']['requested'] += 1
                    counter['corrupted']['requested'] += 1
                    any_requested = True
                    if mode_counter:
                        mode_counter['clean']['requested'] += 1
                        mode_counter['corrupted']['requested'] += 1

                presence_data = ablation.get(field)
                if not isinstance(presence_data, dict):
                    continue

                for twist_type in ['clean', 'corrupted']:
                    twist_entry = presence_data.get(twist_type) or {}
                    present = twist_entry.get('present')
                    confidence = twist_entry.get('confidence')

                    if present is not None:
                        counter[twist_type]['present_total'] += 1
                        counter[twist_type]['parsed'] += 1
                        if present:
                            counter[twist_type]['present_true'] += 1
                        if mode_counter:
                            mode_counter[twist_type]['present_total'] += 1
                            mode_counter[twist_type]['parsed'] += 1
                            if present:
                                mode_counter[twist_type]['present_true'] += 1
                    elif isinstance(confidence, (int, float)):
                        counter[twist_type]['parsed'] += 1
                        if mode_counter:
                            mode_counter[twist_type]['parsed'] += 1

                    if isinstance(confidence, (int, float)):
                        counter[twist_type]['confidence_sum'] += confidence
                        counter[twist_type]['confidence_count'] += 1
                        if mode_counter:
                            mode_counter[twist_type]['confidence_sum'] += confidence
                            mode_counter[twist_type]['confidence_count'] += 1

        if result_threshold is not None:
            thresholds.add(result_threshold)
        if result_batch is not None:
            batch_sizes.add(result_batch)
        if result_max_new is not None:
            max_new_tokens.add(result_max_new)

    if not any_requested:
        return {
            'available': False,
            'models': [],
            'fields': {},
            'thresholds': [],
            'batch_sizes': [],
            'max_new_tokens': [],
            'parse_warnings': total_parse_warnings,
            'overall': {}
        }

    for field_counter in presence_summary.values():
        _finalize_presence_counter(field_counter)
    for mode_counters in presence_by_mode.values():
        for field_counter in mode_counters.values():
            _finalize_presence_counter(field_counter)

    overall = {}
    ablation_story_corrupted = presence_summary['ablation_presence']['corrupted']
    if ablation_story_corrupted['present_total'] > 0:
        overall['ablation_story_corrupted_present_pct'] = ablation_story_corrupted['present_pct']
        overall['ablation_story_corrupted_present_count'] = (
            ablation_story_corrupted['present_true'],
            ablation_story_corrupted['present_total']
        )

    baseline_story_corrupted = presence_summary['baseline_presence']['corrupted']
    if baseline_story_corrupted['present_total'] > 0:
        overall['baseline_story_corrupted_present_pct'] = baseline_story_corrupted['present_pct']
        overall['baseline_story_corrupted_present_count'] = (
            baseline_story_corrupted['present_true'],
            baseline_story_corrupted['present_total']
        )

    overall['parse_warnings'] = total_parse_warnings

    return {
        'available': True,
        'models': sorted(presence_models),
        'fields': presence_summary,
        'by_think_mode': presence_by_mode,
        'thresholds': sorted(thresholds),
        'batch_sizes': sorted(batch_sizes),
        'max_new_tokens': sorted(max_new_tokens),
        'parse_warnings': total_parse_warnings,
        'overall': overall
    }


def analyze_presence_correlation(all_results: List[Dict]) -> Dict:
    """Correlation between semantic similarity shifts and LLM presence confidence."""

    story_sem_diffs = []
    story_conf = []
    story_modes = []

    reasoning_sem_diffs = []
    reasoning_conf = []

    for result in all_results:
        if not result or 'ablations' not in result:
            continue

        think_mode = result.get('think_mode')

        for ablation in result['ablations']:
            # Story correlation
            semantic_clean = ablation.get('ablation_content_vs_clean')
            semantic_corrupt = ablation.get('ablation_content_vs_corrupted')
            if semantic_clean is not None and semantic_corrupt is not None:
                presence = ablation.get('ablation_presence') or {}
                corrupt_presence = presence.get('corrupted') if isinstance(presence, dict) else None
                if corrupt_presence:
                    conf = corrupt_presence.get('confidence')
                    if isinstance(conf, (int, float)):
                        story_sem_diffs.append(semantic_corrupt - semantic_clean)
                        story_conf.append(conf)
                        story_modes.append(think_mode)

            # Reasoning correlation (allow_more_thinking only)
            if think_mode != 'allow_more_thinking':
                continue

            semantic_reasoning_corrupt = ablation.get('ablation_reasoning_vs_corrupted')
            baseline_reasoning_corrupt = ablation.get('baseline_reasoning_vs_corrupted')
            if semantic_reasoning_corrupt is None or baseline_reasoning_corrupt is None:
                continue

            reasoning_presence = ablation.get('ablation_reasoning_presence') or {}
            corrupt_reasoning_presence = reasoning_presence.get('corrupted') if isinstance(reasoning_presence, dict) else None
            if not corrupt_reasoning_presence:
                continue

            conf_reason = corrupt_reasoning_presence.get('confidence')
            if isinstance(conf_reason, (int, float)):
                reasoning_sem_diffs.append(semantic_reasoning_corrupt - baseline_reasoning_corrupt)
                reasoning_conf.append(conf_reason)

    def compute_stats(xs: List[float], ys: List[float]):
        points = len(xs)
        if points < 2:
            return {
                'points': points,
                'correlation': None,
                'mean_semantic': float(np.mean(xs)) if xs else None,
                'mean_confidence': float(np.mean(ys)) if ys else None
            }

        arr_x = np.array(xs, dtype=float)
        arr_y = np.array(ys, dtype=float)
        if np.std(arr_x) == 0 or np.std(arr_y) == 0:
            corr = None
        else:
            corr = float(np.corrcoef(arr_x, arr_y)[0, 1])

        return {
            'points': points,
            'correlation': corr,
            'mean_semantic': float(np.mean(arr_x)),
            'mean_confidence': float(np.mean(arr_y))
        }

    story_stats = compute_stats(story_sem_diffs, story_conf)
    story_stats['think_modes'] = {
        'no_more_thinking': story_modes.count('no_more_thinking'),
        'allow_more_thinking': story_modes.count('allow_more_thinking')
    }

    reasoning_stats = compute_stats(reasoning_sem_diffs, reasoning_conf)

    return {
        'story': story_stats,
        'reasoning': reasoning_stats
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
        if cot['reasoning_prefers_clean_pct'] is not None:
            total_reasoning = cot['reasoning_total_ablations']
            print(
                "  Focuses on clean: "
                f"{cot['reasoning_prefers_clean_count']}"
                f"/{total_reasoning} ({cot['reasoning_prefers_clean_pct']:.1f}%)"
            )
        else:
            print("  Skipped (no reasoning outputs in this batch)")
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

    # 5. Reasoning Shift Correlation
    print("\n5. REASONING SHIFT (allow_more_thinking)")
    print("-" * 80)
    shift = analysis['reasoning_shift']
    if shift['total_points'] < 2 or shift['correlation'] is None:
        print("\nNot enough reasoning outputs to compute correlation.")
    else:
        print(f"\nTotal reasoning comparisons: {shift['total_points']}")
        print(f"Mean baseline vs corrupted similarity: {shift['mean_baseline_similarity']:.3f}")
        sign = '+' if shift['mean_shift'] >= 0 else ''
        print(f"Mean shift toward corrupted twist: {sign}{shift['mean_shift']:.3f}")
        print(f"Pearson correlation (baseline alignment vs shift): {shift['correlation']:.3f}")
        if shift['content_prefers_corrupted_pct'] is not None:
            print(
                "Content follow rate for corrupted twist (same subset): "
                f"{shift['content_prefers_corrupted_pct']:.1f}%"
            )

    # 6. LLM Twist Presence Detection
    presence = analysis.get('presence_detection', {})
    print("\n6. LLM TWIST PRESENCE DETECTION")
    print("-" * 80)
    if not presence or not presence.get('available'):
        print("\nPresence detection not requested in these runs.")
    else:
        models = presence.get('models') or []
        thresholds = presence.get('thresholds') or []
        batch_sizes = presence.get('batch_sizes') or []
        max_new_tokens = presence.get('max_new_tokens') or []

        if models:
            print(f"\nModels: {', '.join(models)}")
        if thresholds:
            print(f"Thresholds: {', '.join(f'{t:.2f}' if isinstance(t, float) else str(t) for t in thresholds)}")
        if batch_sizes:
            print(f"Batch sizes: {', '.join(str(b) for b in batch_sizes)}")
        if max_new_tokens:
            print(f"Max new tokens: {', '.join(str(m) for m in max_new_tokens)}")

        fields = presence.get('fields', {})
        by_mode = presence.get('by_think_mode', {}) or {}

        def print_presence_table(title: str, data_map: Dict[str, Dict]):
            print(f"\n{title}:")
            for field_key, label in PRESENCE_FIELDS.items():
                data = data_map.get(field_key)
                if not data:
                    continue
                print(f"  {label}:")
                for twist_type in ['clean', 'corrupted']:
                    entry = data.get(twist_type, {})
                    requested = entry.get('requested', 0)
                    parsed = entry.get('parsed', 0)
                    present_total = entry.get('present_total', 0)
                    present_true = entry.get('present_true', 0)
                    present_pct = entry.get('present_pct')
                    parsed_pct = entry.get('parsed_pct')
                    confidence_avg = entry.get('confidence_avg')

                    twist_label = '    Clean twist' if twist_type == 'clean' else '    Corrupted twist'
                    present_str = (
                        f"present {present_true}/{present_total} ({present_pct:.1f}%)"
                        if present_total > 0 and present_pct is not None
                        else f"present {present_true}/{present_total}"
                    )
                    parsed_str = (
                        f"parsed {parsed}/{requested} ({parsed_pct:.1f}%)"
                        if requested > 0 and parsed_pct is not None
                        else f"parsed {parsed}/{requested}"
                    )
                    conf_str = (
                        f", avg conf {confidence_avg:.2f}"
                        if confidence_avg is not None else ''
                    )
                    print(f"{twist_label}: {present_str}; {parsed_str}{conf_str}")

        print_presence_table('All think modes combined', fields)

        mode_labels = {
            'no_more_thinking': 'No More Thinking',
            'allow_more_thinking': 'Allow More Thinking'
        }
        for mode_key, label in mode_labels.items():
            mode_data = by_mode.get(mode_key)
            if not mode_data:
                continue
            print_presence_table(f"{label}", mode_data)

        overall = presence.get('overall', {})
        if overall:
            leak = overall.get('ablation_story_corrupted_present_pct')
            if leak is not None:
                count = overall.get('ablation_story_corrupted_present_count')
                if count:
                    print(
                        f"\nAblation story corrupted presence: {leak:.1f}% ({count[0]}/{count[1]})"
                    )
            baseline_leak = overall.get('baseline_story_corrupted_present_pct')
            if baseline_leak is not None:
                count = overall.get('baseline_story_corrupted_present_count')
                if count:
                    print(
                        f"Baseline story corrupted presence: {baseline_leak:.1f}% ({count[0]}/{count[1]})"
                    )

        warnings = presence.get('parse_warnings', 0)
        if warnings:
            print(f"\nPresence parser warnings (aggregate): {warnings}")

    # 7. LLM Presence vs Semantic Correlation
    presence_corr = analysis.get('presence_correlation', {})
    print("\n7. LLM PRESENCE VS SEMANTIC CORRELATION")
    print("-" * 80)
    story_corr = presence_corr.get('story', {})
    if story_corr.get('points', 0) < 2 or story_corr.get('correlation') is None:
        print("\nStory content: not enough data for correlation.")
    else:
        modes = story_corr.get('think_modes', {}) or {}
        print(f"\nStory content points: {story_corr['points']} (no_more_thinking={modes.get('no_more_thinking', 0)}, allow_more_thinking={modes.get('allow_more_thinking', 0)})")
        print(f"Mean semantic diff (corrupted - clean): {story_corr['mean_semantic']:.3f}")
        print(f"Mean LLM confidence (corrupted twist): {story_corr['mean_confidence']:.3f}")
        print(f"Pearson correlation: {story_corr['correlation']:.3f}")

    reasoning_corr = presence_corr.get('reasoning', {})
    if reasoning_corr.get('points', 0) < 2 or reasoning_corr.get('correlation') is None:
        print("\nReasoning (allow_more_thinking): not enough data for correlation.")
    else:
        print("\nReasoning (allow_more_thinking):")
        print(f"  Points: {reasoning_corr['points']}")
        print(f"  Mean semantic shift (Δ corrupted similarity): {reasoning_corr['mean_semantic']:.3f}")
        print(f"  Mean LLM confidence (corrupted twist): {reasoning_corr['mean_confidence']:.3f}")
        print(f"  Pearson correlation: {reasoning_corr['correlation']:.3f}")

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
        'reasoning_shift': analyze_reasoning_shift(all_results),
        'presence_detection': analyze_presence_detection(all_results),
        'presence_correlation': analyze_presence_correlation(all_results),
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
