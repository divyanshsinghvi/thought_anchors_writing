#!/usr/bin/env python3
"""
Compare evaluation scores between no_cot and other CoT variants.
Analyzes scores by evaluator and dimension.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import statistics

from config import OUTPUT_DIR_NON_FICTION, OUTPUT_DIR_FICTION, EVALUATED_SUFFIX


def load_evaluation_file(file_path: Path) -> Dict:
    """Load an evaluation JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_scores_by_evaluator(eval_data: Dict) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract scores organized by evaluator and dimension.

    Returns:
        {evaluator_name: {dimension: [scores]}}
    """
    scores_by_evaluator = defaultdict(lambda: defaultdict(list))

    if 'evaluations' not in eval_data:
        return {}

    for evaluation in eval_data['evaluations']:
        evaluator = evaluation.get('evaluator', 'unknown')
        scores = evaluation.get('evaluation', {}).get('scores', {})

        for dimension, score in scores.items():
            if score is not None:
                scores_by_evaluator[evaluator][dimension].append(score)

    return dict(scores_by_evaluator)


def find_matching_files(base_dir: Path, system_prompt_name: str, prompt_id: str) -> List[Path]:
    """Find evaluated files for a specific system prompt and prompt ID."""
    eval_dir = base_dir / f"{system_prompt_name}{EVALUATED_SUFFIX}"

    if not eval_dir.exists():
        return []

    # Find files matching the prompt_id
    matching_files = list(eval_dir.glob(f"{prompt_id}_*_evaluated.json"))
    return matching_files


def compare_system_prompts(
    base_dir: Path,
    baseline_prompt: str,
    comparison_prompts: List[str],
    prompt_ids: List[str] = None
) -> Dict:
    """
    Compare baseline system prompt (e.g., no_cot) with other system prompts.

    Args:
        base_dir: Base directory (non_fiction or fiction)
        baseline_prompt: Name of baseline system prompt (e.g., "no_cot")
        comparison_prompts: List of system prompt names to compare against
        prompt_ids: Optional list of specific prompt IDs to analyze

    Returns:
        Dictionary with comparison results
    """
    results = {
        'baseline': baseline_prompt,
        'comparisons': {},
        'prompt_results': {}
    }

    # Get all prompt files for baseline
    baseline_dir = base_dir / f"{baseline_prompt}{EVALUATED_SUFFIX}"

    if not baseline_dir.exists():
        print(f"Error: Baseline directory {baseline_dir} does not exist")
        return results

    baseline_files = sorted(baseline_dir.glob("*_evaluated.json"))

    if not baseline_files:
        print(f"Warning: No evaluated files found in {baseline_dir}")
        return results

    # Extract prompt IDs from baseline files
    if prompt_ids is None:
        prompt_ids = []
        for f in baseline_files:
            # Extract prompt_id from filename (e.g., "nf_001_no_cot_evaluated.json" -> "nf_001")
            parts = f.stem.split('_')
            if len(parts) >= 2:
                prompt_id = f"{parts[0]}_{parts[1]}"
                prompt_ids.append(prompt_id)

    print(f"\nAnalyzing {len(prompt_ids)} prompts...")
    print(f"Baseline: {baseline_prompt}")
    print(f"Comparing with: {', '.join(comparison_prompts)}\n")

    # Initialize comparison structure
    for comp_prompt in comparison_prompts:
        results['comparisons'][comp_prompt] = {
            'by_evaluator': defaultdict(lambda: {
                'dimensions': defaultdict(lambda: {
                    'baseline_scores': [],
                    'comparison_scores': [],
                    'differences': []
                })
            }),
            'overall': defaultdict(lambda: {
                'baseline_scores': [],
                'comparison_scores': [],
                'differences': []
            })
        }

    # Process each prompt
    for prompt_id in prompt_ids:
        # Load baseline file
        baseline_files = find_matching_files(base_dir, baseline_prompt, prompt_id)

        if not baseline_files:
            print(f"Warning: No baseline file found for {prompt_id}")
            continue

        baseline_file = baseline_files[0]
        baseline_data = load_evaluation_file(baseline_file)
        baseline_scores = extract_scores_by_evaluator(baseline_data)

        results['prompt_results'][prompt_id] = {
            'baseline': {baseline_prompt: baseline_scores},  # Store with baseline name as key
            'comparisons': {}
        }

        # Compare with each comparison prompt
        for comp_prompt in comparison_prompts:
            comp_files = find_matching_files(base_dir, comp_prompt, prompt_id)

            if not comp_files:
                print(f"Warning: No file found for {prompt_id} with {comp_prompt}")
                continue

            comp_file = comp_files[0]
            comp_data = load_evaluation_file(comp_file)
            comp_scores = extract_scores_by_evaluator(comp_data)

            results['prompt_results'][prompt_id]['comparisons'][comp_prompt] = comp_scores

            # Aggregate scores by evaluator
            for evaluator in baseline_scores.keys():
                if evaluator not in comp_scores:
                    continue

                for dimension in baseline_scores[evaluator].keys():
                    if dimension not in comp_scores[evaluator]:
                        continue

                    baseline_dim_scores = baseline_scores[evaluator][dimension]
                    comp_dim_scores = comp_scores[evaluator][dimension]

                    # Use mean of scores for this prompt/evaluator/dimension
                    baseline_mean = statistics.mean(baseline_dim_scores)
                    comp_mean = statistics.mean(comp_dim_scores)
                    difference = comp_mean - baseline_mean

                    # Store in results
                    results['comparisons'][comp_prompt]['by_evaluator'][evaluator]['dimensions'][dimension]['baseline_scores'].append(baseline_mean)
                    results['comparisons'][comp_prompt]['by_evaluator'][evaluator]['dimensions'][dimension]['comparison_scores'].append(comp_mean)
                    results['comparisons'][comp_prompt]['by_evaluator'][evaluator]['dimensions'][dimension]['differences'].append(difference)

                    # Also store in overall (across all evaluators)
                    results['comparisons'][comp_prompt]['overall'][dimension]['baseline_scores'].append(baseline_mean)
                    results['comparisons'][comp_prompt]['overall'][dimension]['comparison_scores'].append(comp_mean)
                    results['comparisons'][comp_prompt]['overall'][dimension]['differences'].append(difference)

    return results


def print_comparison_report(results: Dict):
    """Print a formatted comparison report."""
    baseline = results['baseline']

    print(f"\n{'='*80}")
    print(f"CoT Comparison Report: {baseline} (baseline) vs. other variants")
    print(f"{'='*80}\n")

    for comp_prompt, comp_data in results['comparisons'].items():
        print(f"\n{'─'*80}")
        print(f"Comparison: {baseline} vs. {comp_prompt}")
        print(f"{'─'*80}\n")

        # Print by evaluator
        print("By Evaluator:")
        print("-" * 80)

        for evaluator, eval_data in comp_data['by_evaluator'].items():
            print(f"\n  Evaluator: {evaluator}")
            print(f"  {'-' * 76}")

            for dimension, dim_data in eval_data['dimensions'].items():
                if not dim_data['differences']:
                    continue

                baseline_mean = statistics.mean(dim_data['baseline_scores'])
                comp_mean = statistics.mean(dim_data['comparison_scores'])
                diff_mean = statistics.mean(dim_data['differences'])
                diff_std = statistics.stdev(dim_data['differences']) if len(dim_data['differences']) > 1 else 0

                # Determine if improvement or decline
                indicator = "↑" if diff_mean > 0 else "↓" if diff_mean < 0 else "="

                print(f"    {dimension:25s}: {baseline_mean:5.2f} → {comp_mean:5.2f}  "
                      f"{indicator} {diff_mean:+5.2f} (±{diff_std:.2f})  "
                      f"[n={len(dim_data['differences'])}]")

        # Print overall (across all evaluators)
        print(f"\n  Overall (all evaluators):")
        print(f"  {'-' * 76}")

        for dimension, dim_data in comp_data['overall'].items():
            if not dim_data['differences']:
                continue

            baseline_mean = statistics.mean(dim_data['baseline_scores'])
            comp_mean = statistics.mean(dim_data['comparison_scores'])
            diff_mean = statistics.mean(dim_data['differences'])
            diff_std = statistics.stdev(dim_data['differences']) if len(dim_data['differences']) > 1 else 0

            indicator = "↑" if diff_mean > 0 else "↓" if diff_mean < 0 else "="

            print(f"    {dimension:25s}: {baseline_mean:5.2f} → {comp_mean:5.2f}  "
                  f"{indicator} {diff_mean:+5.2f} (±{diff_std:.2f})  "
                  f"[n={len(dim_data['differences'])}]")


def print_summary_table(results: Dict):
    """Print a comprehensive summary table comparing all CoT variants."""
    baseline = results['baseline']
    comparisons = results['comparisons']
    prompt_results = results['prompt_results']

    if not comparisons:
        return

    print(f"\n{'='*100}")
    print(f"SUMMARY: Comparison Table (All CoT Variants)")
    print(f"{'='*100}\n")

    # Get all system prompts (baseline + comparisons)
    all_prompts = [baseline] + list(comparisons.keys())

    # Get all evaluators from first comparison
    evaluators = set()
    for comp_data in comparisons.values():
        evaluators.update(comp_data['by_evaluator'].keys())
    evaluators = sorted(evaluators)

    # Get all dimensions
    dimensions = ['factual_accuracy', 'narrative_quality', 'coherence', 'authenticity']

    for evaluator in evaluators:
        print(f"\nEvaluator: {evaluator}")
        print("─" * 100)

        # Create header
        header = f"{'Dimension':<25}"
        for prompt in all_prompts:
            header += f" | {prompt:>12}"
        header += " | Best"
        print(header)
        print("─" * 100)

        # Print each dimension
        for dimension in dimensions:
            # Collect scores for this dimension across all prompts
            scores = {}

            # Baseline scores
            baseline_scores = []
            for prompt_id, prompt_data in prompt_results.items():
                if baseline in prompt_data['baseline'] and evaluator in prompt_data['baseline'][baseline]:
                    if dimension in prompt_data['baseline'][baseline][evaluator]:
                        baseline_scores.extend(prompt_data['baseline'][baseline][evaluator][dimension])

            if baseline_scores:
                scores[baseline] = statistics.mean(baseline_scores)
            else:
                scores[baseline] = None

            # Comparison scores
            for comp_prompt in comparisons.keys():
                comp_scores = []
                for prompt_id, prompt_data in prompt_results.items():
                    if comp_prompt in prompt_data.get('comparisons', {}):
                        if evaluator in prompt_data['comparisons'][comp_prompt]:
                            if dimension in prompt_data['comparisons'][comp_prompt][evaluator]:
                                comp_scores.extend(prompt_data['comparisons'][comp_prompt][evaluator][dimension])

                if comp_scores:
                    scores[comp_prompt] = statistics.mean(comp_scores)
                else:
                    scores[comp_prompt] = None

            # Find best score
            valid_scores = {k: v for k, v in scores.items() if v is not None}
            best_prompt = max(valid_scores, key=valid_scores.get) if valid_scores else None

            # Print row
            row = f"{dimension:<25}"
            for prompt in all_prompts:
                score = scores.get(prompt)
                if score is not None:
                    is_best = (prompt == best_prompt)
                    score_str = f"{score:>5.2f}"
                    if is_best:
                        score_str = f"*{score_str}*"
                    row += f" | {score_str:>12}"
                else:
                    row += f" | {'-':>12}"

            row += f" | {best_prompt if best_prompt else '-'}"
            print(row)

        print()

    # Print per-prompt breakdown
    print(f"\n{'='*100}")
    print(f"PER-PROMPT BREAKDOWN")
    print(f"{'='*100}\n")

    for prompt_id in sorted(prompt_results.keys()):
        print(f"\nPrompt: {prompt_id}")
        print("─" * 100)

        prompt_data = prompt_results[prompt_id]

        for evaluator in evaluators:
            # Check if this evaluator has data for this prompt
            has_baseline_data = (baseline in prompt_data.get('baseline', {}) and
                                evaluator in prompt_data['baseline'].get(baseline, {}))

            # Check if there's any comparison data
            has_comparison_data = False
            for comp_prompt in comparisons.keys():
                if (comp_prompt in prompt_data.get('comparisons', {}) and
                    evaluator in prompt_data['comparisons'].get(comp_prompt, {})):
                    has_comparison_data = True
                    break

            if not has_baseline_data and not has_comparison_data:
                continue

            print(f"  Evaluator: {evaluator}")

            # Create header
            header = f"    {'Dimension':<25}"
            for prompt in all_prompts:
                header += f" | {prompt:>12}"
            header += " | Best"
            print(header)
            print("  " + "─" * 96)

            # Print each dimension
            for dimension in dimensions:
                scores = {}

                # Baseline - fix the access path
                if has_baseline_data:
                    baseline_eval_data = prompt_data['baseline'][baseline].get(evaluator, {})
                    if dimension in baseline_eval_data:
                        dim_scores = baseline_eval_data[dimension]
                        scores[baseline] = statistics.mean(dim_scores) if dim_scores else None

                # Comparisons
                for comp_prompt in comparisons.keys():
                    if comp_prompt in prompt_data.get('comparisons', {}):
                        comp_eval_data = prompt_data['comparisons'][comp_prompt].get(evaluator, {})
                        if dimension in comp_eval_data:
                            dim_scores = comp_eval_data[dimension]
                            scores[comp_prompt] = statistics.mean(dim_scores) if dim_scores else None

                # Skip if no scores available
                if not scores:
                    continue

                # Find best
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                best_prompt = max(valid_scores, key=valid_scores.get) if valid_scores else None

                # Print row
                row = f"    {dimension:<25}"
                for prompt in all_prompts:
                    score = scores.get(prompt)
                    if score is not None:
                        is_best = (prompt == best_prompt)
                        score_str = f"{score:>5.2f}"
                        if is_best:
                            score_str = f"*{score_str}*"
                        row += f" | {score_str:>12}"
                    else:
                        row += f" | {'-':>12}"

                row += f" | {best_prompt if best_prompt else '-'}"
                print(row)

            print()


def save_comparison_json(results: Dict, output_file: Path):
    """Save detailed comparison results to JSON."""
    # Convert defaultdicts to regular dicts for JSON serialization
    def convert_to_dict(obj):
        if isinstance(obj, defaultdict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        return obj

    results_serializable = convert_to_dict(results)

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare evaluation scores between no_cot and other CoT variants'
    )
    parser.add_argument('--type', type=str, choices=['fiction', 'non-fiction'],
                        default='non-fiction',
                        help='Type of samples to analyze')
    parser.add_argument('--baseline', type=str, default='no_cot',
                        help='Baseline system prompt to compare against')
    parser.add_argument('--compare', type=str, nargs='+',
                        default=['nohint_cot', 'minimal_cot', 'freeform_cot', 'structured_cot'],
                        help='System prompts to compare with baseline')
    parser.add_argument('--prompt-ids', type=str, nargs='+',
                        help='Specific prompt IDs to analyze (default: all)')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for detailed results')

    args = parser.parse_args()

    # Get base directory
    base_dir = OUTPUT_DIR_FICTION if args.type == 'fiction' else OUTPUT_DIR_NON_FICTION

    # Run comparison
    results = compare_system_prompts(
        base_dir,
        args.baseline,
        args.compare,
        args.prompt_ids
    )

    # Print detailed report
    print_comparison_report(results)

    # Print summary table
    print_summary_table(results)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        save_comparison_json(results, output_path)

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
