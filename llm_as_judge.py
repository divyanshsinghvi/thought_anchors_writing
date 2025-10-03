#!/usr/bin/env python3
"""
Evaluate generated non-fiction stories using LLM-as-judge.
Supports multiple evaluator models and multiple samples per evaluation to reduce bias.
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv
from rollouts import RolloutsClient
from config import (
    OUTPUT_DIR_NON_FICTION,
    OUTPUT_DIR_FICTION,
    BASE_DIR,
    CACHE_PATH,
    PROMPTS_YAML,
    FICTIONAL_PROMPTS_YAML,
    DEFAULT_EVALUATOR_MODELS,
    DEFAULT_EVALUATION_TEMPERATURE,
    DEFAULT_EVALUATION_SAMPLES,
    EVALUATED_SUFFIX
)

load_dotenv()


def get_system_prompt_names(prompt_type: str) -> List[str]:
    """Load system prompt names from the YAML file."""
    import yaml

    yaml_path = PROMPTS_YAML if prompt_type == 'non-fiction' else FICTIONAL_PROMPTS_YAML

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Handle both old and new formats
    # New format - extract all system prompt names
    return list(data.get('system_prompts', {}).keys())


# Evaluation dimensions for non-fiction
EVALUATION_DIMENSIONS = {
    "factual_accuracy": {
        "description": "Are the facts correct? No fabricated or incorrect details?",
        "weight": 1.0
    },
    "narrative_quality": {
        "description": "Is it engaging, well-written, and compelling?",
        "weight": 1.0
    },
    "coherence": {
        "description": "Does it flow logically from start to finish?",
        "weight": 1.0
    },
    "authenticity": {
        "description": "Does it feel true to the subject matter and period?",
        "weight": 1.0
    }
}

# Evaluation prompt template
EVAL_PROMPT_TEMPLATE = """You are an expert evaluator of non-fiction writing. Evaluate the following story on specific dimensions.

**Story to Evaluate:**
{story}

**Evaluation Task:**
Rate this story on a scale of 1-10 for each dimension below. Be critical but fair.

**Dimensions:**

1. **Factual Accuracy** (1-10): Are the facts correct? No fabricated or incorrect details?
   - 1-3: Major factual errors or fabrications
   - 4-6: Some inaccuracies or questionable claims
   - 7-8: Mostly accurate with minor issues
   - 9-10: Highly accurate and well-researched

2. **Narrative Quality** (1-10): Is it engaging, well-written, and compelling?
   - 1-3: Poorly written, boring, hard to follow
   - 4-6: Adequate but unremarkable writing
   - 7-8: Well-written and engaging
   - 9-10: Exceptional prose and storytelling

3. **Coherence** (1-10): Does it flow logically from start to finish?
   - 1-3: Disjointed, confusing structure
   - 4-6: Some organizational issues
   - 7-8: Clear, logical flow
   - 9-10: Masterfully structured

4. **Authenticity** (1-10): Does it feel true to the subject matter and period?
   - 1-3: Anachronistic or inauthentic
   - 4-6: Generic or somewhat artificial
   - 7-8: Believable and well-contextualized
   - 9-10: Deeply authentic and immersive

**Response Format:**
Provide your evaluation in this EXACT format:

FACTUAL_ACCURACY: [score]
REASONING: [2-3 sentences explaining the score]

NARRATIVE_QUALITY: [score]
REASONING: [2-3 sentences explaining the score]

COHERENCE: [score]
REASONING: [2-3 sentences explaining the score]

AUTHENTICITY: [score]
REASONING: [2-3 sentences explaining the score]

OVERALL_ASSESSMENT: [Brief 2-3 sentence summary of strengths and weaknesses]

**Important Guidelines:**
- Be consistent across evaluations
- Don't favor longer stories
- Don't favor complex vocabulary over clarity
- Focus on the actual content and craft
- Be honest about weaknesses while acknowledging strengths
"""


def parse_evaluation_response(response_text: str) -> Dict[str, any]:
    """Parse the evaluation response into structured scores and reasoning."""
    scores = {}
    reasoning = {}
    
    # Extract scores
    dimensions = ['FACTUAL_ACCURACY', 'NARRATIVE_QUALITY', 'COHERENCE', 'AUTHENTICITY']
    
    for dimension in dimensions:
        # Find score
        score_pattern = f"{dimension}:\\s*(\\d+(?:\\.\\d+)?)"
        score_match = re.search(score_pattern, response_text, re.IGNORECASE)
        
        if score_match:
            scores[dimension.lower()] = float(score_match.group(1))
        else:
            scores[dimension.lower()] = None
            print(f"⚠️  Warning: Could not extract {dimension} score")
        
        # Find reasoning (text between current dimension and next dimension or OVERALL)
        reasoning_pattern = f"{dimension}:.*?REASONING:\\s*(.*?)(?=(?:NARRATIVE_QUALITY|COHERENCE|AUTHENTICITY|OVERALL_ASSESSMENT|$))"
        reasoning_match = re.search(reasoning_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if reasoning_match:
            reasoning[dimension.lower()] = reasoning_match.group(1).strip()
        else:
            reasoning[dimension.lower()] = ""
    
    # Extract overall assessment
    overall_pattern = r"OVERALL_ASSESSMENT:\s*(.*?)(?:\n\n|$)"
    overall_match = re.search(overall_pattern, response_text, re.DOTALL | re.IGNORECASE)
    overall_assessment = overall_match.group(1).strip() if overall_match else ""
    
    return {
        "scores": scores,
        "reasoning": reasoning,
        "overall_assessment": overall_assessment,
        "raw_response": response_text
    }


def extract_story_from_response(response_data: Dict) -> str:
    """Extract the actual story text from the generation response."""
    # Handle different response formats
    if isinstance(response_data, dict):
        # Check if this is a Rollouts dict with responses array
        if 'responses' in response_data and isinstance(response_data['responses'], list):
            num_responses = len(response_data['responses'])

            if num_responses == 0:
                raise ValueError("Response data contains empty responses array")
            elif num_responses > 1:
                raise ValueError(
                    f"Expected single response for story generation, but found {num_responses} responses. "
                    f"This sample may have been generated with n_samples > 1."
                )

            first_response = response_data['responses'][0]
            # Use 'content' field which has post-reasoning text (the actual story)
            full_text = first_response.get('content', first_response.get('full', ''))
        # Check for direct 'content' field
        elif 'content' in response_data:
            full_text = response_data['content']
        # Check for 'text' field (common in rollouts)
        elif 'text' in response_data:
            full_text = response_data['text']
        elif 'full' in response_data:
            full_text = response_data['full']
        else:
            # Try to find any text field
            full_text = str(response_data)
    else:
        full_text = str(response_data)

    # Try to extract story from XML tags
    story_match = re.search(r'<story>(.*?)</story>', full_text, re.DOTALL)
    if story_match:
        return story_match.group(1).strip()

    # If no XML tags, return the full text
    return full_text


async def evaluate_sample_async(
    client: RolloutsClient,
    sample_data: Dict,
    evaluator_name: str,
    n_samples: int = 1
) -> List[Dict]:
    """Evaluate a single sample using LLM-as-judge with n_samples (async)."""

    # Extract story
    story = extract_story_from_response(sample_data.get('response', {}))

    # Create evaluation prompt
    eval_prompt = EVAL_PROMPT_TEMPLATE.format(story=story)

    # Generate n_samples evaluations
    print(f"  Evaluating {sample_data.get('prompt_id', 'unknown')} with {evaluator_name} (n={n_samples})...")

    response = await client.agenerate(prompt=eval_prompt, n_samples=n_samples)

    # Parse all evaluations - response is a Rollouts object with responses attribute
    evaluations = []
    for idx, rollout_response in enumerate(response.responses):
        result_text = rollout_response.full  # Use 'full' attribute for complete response
        evaluation = parse_evaluation_response(result_text)
        evaluations.append({
            "evaluator": evaluator_name,
            "sample_index": idx,
            "evaluation": evaluation,
            "raw_rollout": rollout_response.to_dict() if hasattr(rollout_response, 'to_dict') else None
        })

    return evaluations


def evaluate_sample_sync(
    client: RolloutsClient,
    sample_data: Dict,
    evaluator_name: str,
    n_samples: int = 1
) -> List[Dict]:
    """Evaluate a single sample using LLM-as-judge with n_samples (sync)."""

    # Extract story
    story = extract_story_from_response(sample_data.get('response', {}))

    # Create evaluation prompt
    eval_prompt = EVAL_PROMPT_TEMPLATE.format(story=story)

    # Generate n_samples evaluations
    print(f"  Evaluating {sample_data.get('prompt_id', 'unknown')} with {evaluator_name} (n={n_samples})...")

    response = client.generate(prompt=eval_prompt, n_samples=n_samples)

    # Parse all evaluations - response is a Rollouts object with responses attribute
    evaluations = []
    for idx, rollout_response in enumerate(response.responses):
        result_text = rollout_response.full  # Use 'full' attribute for complete response
        evaluation = parse_evaluation_response(result_text)
        evaluations.append({
            "evaluator": evaluator_name,
            "sample_index": idx,
            "evaluation": evaluation,
            "raw_rollout": rollout_response.to_dict() if hasattr(rollout_response, 'to_dict') else None
        })

    return evaluations


def load_sample(file_path: Path) -> Dict:
    """Load a sample JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_evaluation(sample_data: Dict, evaluations: List[Dict], output_path: Path):
    """Save evaluation results alongside sample data."""
    output_data = {
        **sample_data,
        "evaluations": evaluations,
        "aggregated_scores": aggregate_scores(evaluations)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved evaluation to: {output_path}")


def aggregate_scores(evaluations: List[Dict]) -> Dict:
    """Aggregate scores from multiple evaluators and samples."""
    if not evaluations:
        return {}
    
    import statistics
    
    aggregated = {
        "mean_scores": {},
        "std_scores": {},
        "all_scores": {},
        "by_evaluator": {},
        "n_evaluations": len(evaluations)
    }
    
    dimensions = ['factual_accuracy', 'narrative_quality', 'coherence', 'authenticity']
    
    # Aggregate across all evaluations
    for dimension in dimensions:
        scores = []
        for eval_data in evaluations:
            score = eval_data['evaluation']['scores'].get(dimension)
            if score is not None:
                scores.append(score)
        
        if scores:
            aggregated['mean_scores'][dimension] = statistics.mean(scores)
            aggregated['std_scores'][dimension] = statistics.stdev(scores) if len(scores) > 1 else 0.0
            aggregated['all_scores'][dimension] = scores
    
    # Aggregate by evaluator (if multiple evaluators)
    evaluators = set(eval_data['evaluator'] for eval_data in evaluations)
    
    for evaluator in evaluators:
        eval_scores = [e for e in evaluations if e['evaluator'] == evaluator]
        
        evaluator_agg = {}
        for dimension in dimensions:
            scores = [e['evaluation']['scores'].get(dimension) 
                     for e in eval_scores 
                     if e['evaluation']['scores'].get(dimension) is not None]
            
            if scores:
                evaluator_agg[dimension] = {
                    'mean': statistics.mean(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'n': len(scores)
                }
        
        aggregated['by_evaluator'][evaluator] = evaluator_agg
    
    # Overall mean
    if aggregated['mean_scores']:
        aggregated['overall_mean'] = statistics.mean(aggregated['mean_scores'].values())
        aggregated['overall_std'] = statistics.stdev(aggregated['mean_scores'].values()) if len(aggregated['mean_scores']) > 1 else 0.0
    
    return aggregated


async def evaluate_single_file_async(
    file_path: Path,
    output_dir: Path,
    evaluator_models: List[str],
    n_samples: int,
    temperature: float
):
    """Evaluate a single file (async)."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample
    sample_data = load_sample(file_path)
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {file_path.name}")
    print(f"Evaluators: {', '.join(evaluator_models)}")
    print(f"Samples per evaluator: {n_samples}")
    print(f"{'='*80}\n")
    
    # Create clients for each evaluator
    clients = {
        model: RolloutsClient(
            model=model,
            temperature=temperature,
            cache_dir=str(CACHE_PATH)
        )
        for model in evaluator_models
    }
    
    # Evaluate with all evaluators
    tasks = []
    for model_name, client in clients.items():
        tasks.append(evaluate_sample_async(client, sample_data, model_name, n_samples))
    
    all_evaluations_nested = await asyncio.gather(*tasks)
    
    # Flatten evaluations
    all_evaluations = []
    for eval_list in all_evaluations_nested:
        all_evaluations.extend(eval_list)
    
    # Save results
    output_path = output_dir / f"{file_path.stem}_evaluated.json"
    save_evaluation(sample_data, all_evaluations, output_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary:")
    print(f"  Total evaluations: {len(all_evaluations)}")
    
    agg = aggregate_scores(all_evaluations)
    if agg.get('mean_scores'):
        print(f"  Overall mean score: {agg['overall_mean']:.2f} ± {agg['overall_std']:.2f}")
        print(f"\n  Dimension scores:")
        for dim, score in agg['mean_scores'].items():
            print(f"    {dim:20s}: {score:.2f} ± {agg['std_scores'][dim]:.2f}")
    print(f"{'='*80}\n")


def evaluate_single_file_sync(
    file_path: Path,
    output_dir: Path,
    evaluator_models: List[str],
    n_samples: int,
    temperature: float
):
    """Evaluate a single file (sync)."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample
    sample_data = load_sample(file_path)
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {file_path.name}")
    print(f"Evaluators: {', '.join(evaluator_models)}")
    print(f"Samples per evaluator: {n_samples}")
    print(f"{'='*80}\n")
    
    # Create clients for each evaluator
    clients = {
        model: RolloutsClient(
            model=model,
            temperature=temperature,
            cache_dir=str(CACHE_PATH)
        )
        for model in evaluator_models
    }
    
    # Evaluate with all evaluators
    all_evaluations = []
    for model_name, client in clients.items():
        evaluations = evaluate_sample_sync(client, sample_data, model_name, n_samples)
        all_evaluations.extend(evaluations)
    
    # Save results
    output_path = output_dir / f"{file_path.stem}_evaluated.json"
    save_evaluation(sample_data, all_evaluations, output_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary:")
    print(f"  Total evaluations: {len(all_evaluations)}")
    
    agg = aggregate_scores(all_evaluations)
    if agg.get('mean_scores'):
        print(f"  Overall mean score: {agg['overall_mean']:.2f} ± {agg['overall_std']:.2f}")
        print(f"\n  Dimension scores:")
        for dim, score in agg['mean_scores'].items():
            print(f"    {dim:20s}: {score:.2f} ± {agg['std_scores'][dim]:.2f}")
    print(f"{'='*80}\n")


async def evaluate_directory_async(
    input_dir: Path,
    output_dir: Path,
    evaluator_models: List[str],
    n_samples: int,
    temperature: float
):
    """Evaluate all samples in a directory (async)."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    sample_files = sorted(input_dir.glob("*.json"))
    
    if not sample_files:
        print(f"No samples found in {input_dir}")
        return
    
    print(f"Found {len(sample_files)} samples to evaluate")
    print(f"Using {len(evaluator_models)} evaluator(s): {', '.join(evaluator_models)}")
    print(f"Samples per evaluator: {n_samples}")
    print(f"Total evaluations per sample: {len(evaluator_models) * n_samples}")
    
    # Create clients for each evaluator
    clients = {
        model: RolloutsClient(
            model=model,
            temperature=temperature,
            cache_dir=str(CACHE_PATH)
        )
        for model in evaluator_models
    }
    
    # Evaluate all samples
    tasks = []
    for sample_file in sample_files:
        sample_data = load_sample(sample_file)
        
        for model_name, client in clients.items():
            tasks.append(
                evaluate_sample_async(client, sample_data, model_name, n_samples)
            )
    
    print(f"\nEvaluating {len(tasks)} total evaluation batches...")
    all_evaluations_nested = await asyncio.gather(*tasks)
    
    # Group evaluations by sample
    evaluations_by_sample = {}
    task_idx = 0
    for sample_file in sample_files:
        sample_id = sample_file.stem
        sample_evals = []
        
        for _ in evaluator_models:
            sample_evals.extend(all_evaluations_nested[task_idx])
            task_idx += 1
        
        evaluations_by_sample[sample_id] = sample_evals
    
    # Save results
    for sample_file in sample_files:
        sample_data = load_sample(sample_file)
        sample_id = sample_file.stem
        
        output_path = output_dir / f"{sample_id}_evaluated.json"
        save_evaluation(sample_data, evaluations_by_sample[sample_id], output_path)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"{'='*80}\n")


def evaluate_directory_sync(
    input_dir: Path,
    output_dir: Path,
    evaluator_models: List[str],
    n_samples: int,
    temperature: float
):
    """Evaluate all samples in a directory (sync)."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    sample_files = sorted(input_dir.glob("*.json"))
    
    if not sample_files:
        print(f"No samples found in {input_dir}")
        return
    
    print(f"Found {len(sample_files)} samples to evaluate")
    print(f"Using {len(evaluator_models)} evaluator(s): {', '.join(evaluator_models)}")
    print(f"Samples per evaluator: {n_samples}")
    print(f"Total evaluations per sample: {len(evaluator_models) * n_samples}")
    
    # Create clients for each evaluator
    clients = {
        model: RolloutsClient(
            model=model,
            temperature=temperature,
            cache_dir=str(CACHE_PATH)
        )
        for model in evaluator_models
    }
    
    total_evaluations = len(sample_files) * len(evaluator_models)
    current = 0
    
    # Evaluate each sample
    for sample_file in sample_files:
        print(f"\n{'='*80}")
        print(f"Evaluating: {sample_file.name}")
        print(f"{'='*80}")
        
        sample_data = load_sample(sample_file)
        all_evaluations = []
        
        for model_name, client in clients.items():
            evaluations = evaluate_sample_sync(client, sample_data, model_name, n_samples)
            all_evaluations.extend(evaluations)
            
            current += 1
            print(f"Progress: {current}/{total_evaluations} evaluation batches completed")
        
        # Save results
        output_path = output_dir / f"{sample_file.stem}_evaluated.json"
        save_evaluation(sample_data, all_evaluations, output_path)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate generated stories using LLM-as-judge',
        epilog=f'Default evaluators: {", ".join(DEFAULT_EVALUATOR_MODELS)}'
    )
    parser.add_argument('--type', type=str, choices=['fiction', 'non-fiction'], required=True,
                        help='Type of samples to evaluate')
    parser.add_argument('--system-prompt', type=str,
                        help='Which system prompt outputs to evaluate (for directory mode). Use "all" for all prompts.')
    parser.add_argument('--file', type=str, nargs='+',
                        help='Path(s) to file(s) to evaluate (overrides directory mode). Can specify multiple files.')
    parser.add_argument('--evaluators', type=str, nargs='+',
                        default=DEFAULT_EVALUATOR_MODELS,
                        help='Evaluator model(s) to use. Multiple models reduce bias.')
    parser.add_argument('--n-samples', type=int, default=DEFAULT_EVALUATION_SAMPLES,
                        help='Number of evaluation samples to generate per evaluator (for variance measurement)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_EVALUATION_TEMPERATURE,
                        help='Temperature for evaluator (lower = more consistent)')
    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use async mode for concurrent evaluation')
    
    args = parser.parse_args()

    # File mode (single or multiple files)
    if args.file:
        file_paths = [Path(f) for f in args.file]

        # Validate all files exist
        for file_path in file_paths:
            if not file_path.exists():
                print(f"Error: File {file_path} does not exist")
                return

        system_prompt_names = get_system_prompt_names(args.type)

        print(f"\n{'='*80}")
        print(f"Evaluating {len(file_paths)} file(s)")
        print(f"{'='*80}\n")

        # Process each file
        for idx, file_path in enumerate(file_paths, 1):
            print(f"\n[{idx}/{len(file_paths)}] Processing: {file_path.name}")

            # Determine output directory based on file location
            # Check if file path contains any known system prompt name
            output_dir = None

            for sys_prompt_name in system_prompt_names:
                if sys_prompt_name in str(file_path):
                    output_dir = file_path.parent.parent / f"{sys_prompt_name}{EVALUATED_SUFFIX}"
                    break

            if not output_dir:
                output_dir = file_path.parent / EVALUATED_SUFFIX.lstrip('_')

            if args.use_async:
                asyncio.run(evaluate_single_file_async(
                    file_path,
                    output_dir,
                    args.evaluators,
                    args.n_samples,
                    args.temperature
                ))
            else:
                evaluate_single_file_sync(
                    file_path,
                    output_dir,
                    args.evaluators,
                    args.n_samples,
                    args.temperature
                )

        print(f"\n{'='*80}")
        print(f"All {len(file_paths)} file(s) evaluated successfully!")
        print(f"{'='*80}\n")

        return
    
    # Directory mode
    if not args.system_prompt:
        print("Error: --system-prompt required for directory mode (or use --file for single file)")
        return
    
    # Get base directory
    if args.type == 'fiction':
        base_dir = OUTPUT_DIR_FICTION
    else:
        base_dir = OUTPUT_DIR_NON_FICTION
    
    # Determine which directories to evaluate
    available_system_prompts = get_system_prompt_names(args.type)

    if args.system_prompt == 'all':
        system_prompts = available_system_prompts
    else:
        if args.system_prompt not in available_system_prompts:
            print(f"Error: Unknown system prompt '{args.system_prompt}'")
            print(f"Available prompts: {', '.join(available_system_prompts)}")
            return
        system_prompts = [args.system_prompt]

    # Evaluate each system prompt directory
    for sys_prompt in system_prompts:
        input_dir = base_dir / sys_prompt
        output_dir = base_dir / f"{sys_prompt}{EVALUATED_SUFFIX}"
        
        if not input_dir.exists():
            print(f"⚠️  Warning: {input_dir} does not exist, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating samples from: {sys_prompt}")
        print(f"{'='*80}\n")
        
        if args.use_async:
            asyncio.run(evaluate_directory_async(
                input_dir, 
                output_dir, 
                args.evaluators,
                args.n_samples,
                args.temperature
            ))
        else:
            evaluate_directory_sync(
                input_dir,
                output_dir,
                args.evaluators,
                args.n_samples,
                args.temperature
            )


if __name__ == "__main__":
    main()