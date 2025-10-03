#!/usr/bin/env python3
"""
Paragraph Ablation Study for Chain-of-Thought Reasoning

This script performs three types of ablation on reasoning traces and generates new rollouts:

Ablation Strategies:
1. keep_later: Remove paragraph N, keep all paragraphs after it
2. remove_after: Remove paragraph N and everything after it (progressive truncation)
3. random_sample: Keep random subset of paragraphs (para_index determines how many to keep)

Think Token Modes:
1. no_more_thinking: Close <think> tag after ablated reasoning - no more thinking allowed
   Format: [System prompt]\n[Writing prompt]\n<think>\n[ablated reasoning]\n</think>

2. allow_more_thinking: Keep <think> tag open - model can continue thinking
   Format: [System prompt]\n[Writing prompt]\n<think>\n[ablated reasoning]

Usage:
    # Single file
    python scripts/paragraph_ablation.py <input_json> --output-dir <dir> --strategy all --think-mode both --num-rollouts 100

    # Multiple files (glob pattern)
    python scripts/paragraph_ablation.py "/path/to/*.json" --output-dir <dir> --strategy all --think-mode both --num-rollouts 100

    # Directory (process all JSON files)
    python scripts/paragraph_ablation.py /path/to/directory/ --output-dir <dir> --strategy all --think-mode both --num-rollouts 100

    # Examples
    python scripts/paragraph_ablation.py /mnt/d/.../fic_001_descriptive_prompt.json --output-dir ablation_rollouts/ --strategy all --think-mode both --num-rollouts 100
    python scripts/paragraph_ablation.py "/mnt/d/.../data/fiction/descriptive_prompt/*.json" --output-dir ablation_rollouts/ --strategy all --think-mode both --num-rollouts 100
    python scripts/paragraph_ablation.py /mnt/d/.../data/fiction/descriptive_prompt/ --output-dir ablation_rollouts/ --strategy all --think-mode both --num-rollouts 100
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import sys
import random
import glob as glob_module

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts import RolloutsClient
from config import MODEL_NAME, CACHE_PATH

# Load environment variables
load_dotenv()


def split_reasoning_into_paragraphs(reasoning: str) -> List[str]:
    """Split reasoning text into paragraphs using double newlines."""
    if not reasoning:
        return []
    paragraphs = [p.strip() for p in reasoning.split('\n\n') if p.strip()]
    return paragraphs


def reconstruct_reasoning(paragraphs: List[str]) -> str:
    """Reconstruct reasoning text from list of paragraphs."""
    return '\n\n'.join(paragraphs)


def ablate_paragraph_keep_later(paragraphs: List[str], para_index: int) -> str:
    """Strategy 1: Remove paragraph at para_index, keep all paragraphs after it."""
    if para_index < 0 or para_index >= len(paragraphs):
        raise ValueError(f"Invalid paragraph index {para_index} for {len(paragraphs)} paragraphs")

    ablated_paragraphs = paragraphs[:para_index] + paragraphs[para_index+1:]
    return reconstruct_reasoning(ablated_paragraphs)


def ablate_paragraph_remove_after(paragraphs: List[str], para_index: int) -> str:
    """Strategy 2: Remove paragraph at para_index and everything after it."""
    if para_index < 0 or para_index >= len(paragraphs):
        raise ValueError(f"Invalid paragraph index {para_index} for {len(paragraphs)} paragraphs")

    ablated_paragraphs = paragraphs[:para_index]
    return reconstruct_reasoning(ablated_paragraphs)


def ablate_random_sample(paragraphs: List[str], para_index: int, seed: int = 42) -> str:
    """Strategy 3: Keep random subset of paragraphs.

    Uses para_index to determine how many paragraphs to keep (para_index + 1).
    For reproducibility, uses seed + para_index as the random seed.
    """
    if para_index < 0 or para_index >= len(paragraphs):
        raise ValueError(f"Invalid paragraph index {para_index} for {len(paragraphs)} paragraphs")

    # Number to keep is para_index + 1 (so para_index=0 keeps 1, para_index=5 keeps 6, etc.)
    num_to_keep = para_index + 1
    if num_to_keep > len(paragraphs):
        num_to_keep = len(paragraphs)

    # Set seed for reproducibility
    random.seed(seed + para_index)

    # Sample random indices and keep them sorted to maintain order
    indices = sorted(random.sample(range(len(paragraphs)), num_to_keep))
    ablated_paragraphs = [paragraphs[i] for i in indices]

    return reconstruct_reasoning(ablated_paragraphs)


def create_ablated_prompt(
    original_data: Dict,
    ablated_reasoning: str,
    strategy: str,
    para_index: int,
    total_paragraphs: int,
    think_mode: str = 'no_more_thinking'
) -> str:
    """
    Create a new prompt with ablated reasoning prepended as context.

    Think modes:
    - no_more_thinking: Close <think> tag after ablated reasoning
      [System prompt]\n[Writing prompt]\n<think>\n[ablated reasoning]\n</think>

    - allow_more_thinking: Keep <think> tag open for model to continue thinking
      [System prompt]\n[Writing prompt]\n<think>\n[ablated reasoning]
    """
    system_prompt = original_data['combined_prompt']

    # Create the prompt with ablated reasoning as context
    if ablated_reasoning.strip():
        if think_mode == 'no_more_thinking':
            ablated_prompt = f"{system_prompt}\n\n<think>\n{ablated_reasoning}\n</think>"
        else:  # allow_more_thinking
            ablated_prompt = f"{system_prompt}\n\n<think>\n{ablated_reasoning}\n"
    else:
        # If no reasoning remains, use system prompt with minimal thinking starter
        if think_mode == 'no_more_thinking':
            ablated_prompt = f"{system_prompt}\n\n<think>\n</think>"
        else:  # allow_more_thinking
            ablated_prompt = f"{system_prompt}\n\n<think>\n"

    return ablated_prompt


async def generate_ablation_rollouts(
    client: RolloutsClient,
    ablated_prompt: str,
    original_data: Dict,
    strategy: str,
    para_index: int,
    total_paragraphs: int,
    num_rollouts: int,
    output_dir: Path,
    think_mode: str
) -> None:
    """Generate rollouts for a single ablation and save each rollout separately.

    Directory structure: ablate_para_N/strategy/think_mode/rollout_NNN.json
    """

    prompt_id = original_data['prompt_id']

    print(f"  Generating {num_rollouts} rollouts for paragraph {para_index} (strategy: {strategy}, think: {think_mode})...")

    # Generate rollouts
    response = await client.agenerate(prompt=ablated_prompt, n_samples=num_rollouts)

    # Extract ablated reasoning for metadata
    ablated_reasoning = ablated_prompt.split('<think>')[-1].split('</think>')[0].strip() if '<think>' in ablated_prompt else ''

    # Create base metadata that's common to all rollouts
    base_metadata = {
        'prompt_id': prompt_id,
        'prompt_text': original_data['prompt_text'],
        'prompt_metadata': original_data['prompt_metadata'],
        'system_prompt_name': original_data['system_prompt_name'],
        'ablation_metadata': {
            'strategy': strategy,
            'think_mode': think_mode,
            'ablated_paragraph_index': para_index,
            'total_original_paragraphs': total_paragraphs,
            'remaining_paragraphs': len(split_reasoning_into_paragraphs(ablated_reasoning)),
            'original_file': str(original_data.get('original_file', 'unknown'))
        },
        'ablated_reasoning': ablated_reasoning,
        'ablated_prompt': ablated_prompt,
        'original_reasoning': original_data['response']['responses'][0]['reasoning'],
        'original_content': original_data['response']['responses'][0]['content'],
        'model': MODEL_NAME
    }

    # Save each rollout separately
    response_dict = response.to_dict() if hasattr(response, 'to_dict') else response
    rollouts = response_dict.get('responses', [])

    for rollout_idx, rollout_data in enumerate(rollouts):
        # Create individual rollout data
        rollout_output = {
            **base_metadata,
            'rollout_index': rollout_idx,
            'rollout_data': rollout_data
        }

        # Create output file: ablate_para_N/strategy/think_mode/rollout_NNN.json
        rollout_file = output_dir / f"rollout_{rollout_idx:03d}.json"
        with open(rollout_file, 'w', encoding='utf-8') as f:
            json.dump(rollout_output, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Saved {len(rollouts)} rollouts to: {output_dir}/")


async def process_ablation_async(
    input_file: Path,
    output_dir: Path,
    strategy: str,
    think_mode: str,
    num_rollouts: int,
    max_concurrent_ablations: int = 0
) -> None:
    """Process ablation for a single file with async rollout generation.

    Args:
        max_concurrent_ablations: Max concurrent ablation tasks (0 = unlimited, default)
    """

    # Load original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract reasoning
    try:
        original_reasoning = data['response']['responses'][0]['reasoning']
    except (KeyError, IndexError) as e:
        raise ValueError(f"Could not extract reasoning from {input_file}: {e}")

    # Split into paragraphs
    paragraphs = split_reasoning_into_paragraphs(original_reasoning)
    total_paragraphs = len(paragraphs)

    print(f"\nProcessing: {input_file.name}")
    print(f"Original reasoning has {total_paragraphs} paragraphs")
    print(f"Will generate {num_rollouts} rollouts per ablation\n")

    # Create rollouts client
    client = RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
        cache_dir=str(CACHE_PATH)
    )

    # Determine strategies to run
    if strategy == 'all':
        strategies_to_run = ['keep_later', 'remove_after', 'random_sample']
    else:
        strategies_to_run = [strategy]

    # Determine think modes to run
    if think_mode == 'both':
        think_modes_to_run = ['no_more_thinking', 'allow_more_thinking']
    else:
        think_modes_to_run = [think_mode]

    # Generate rollouts for each strategy, think mode, and paragraph
    for strat in strategies_to_run:
        for t_mode in think_modes_to_run:
            print(f"Strategy: {strat}, Think mode: {t_mode}\n")

            tasks = []
            for para_idx in range(total_paragraphs):
                # Create directory: output_dir/prompt_id/ablate_para_N/strategy/think_mode/
                rollout_dir = output_dir / data['prompt_id'] / f"ablate_para_{para_idx:02d}" / strat / t_mode
                rollout_dir.mkdir(parents=True, exist_ok=True)

                # Create ablated reasoning based on strategy
                if strat == 'keep_later':
                    ablated_reasoning = ablate_paragraph_keep_later(paragraphs, para_idx)
                elif strat == 'remove_after':
                    ablated_reasoning = ablate_paragraph_remove_after(paragraphs, para_idx)
                elif strat == 'random_sample':
                    ablated_reasoning = ablate_random_sample(paragraphs, para_idx)

                # Create ablated prompt with think mode
                ablated_prompt = create_ablated_prompt(
                    data,
                    ablated_reasoning,
                    strat,
                    para_idx,
                    total_paragraphs,
                    t_mode
                )

                # Queue rollout generation
                task = generate_ablation_rollouts(
                    client,
                    ablated_prompt,
                    data,
                    strat,
                    para_idx,
                    total_paragraphs,
                    num_rollouts,
                    rollout_dir,
                    t_mode
                )
                tasks.append(task)

            # Generate all rollouts for this strategy+think_mode
            if max_concurrent_ablations > 0:
                # Process in batches to limit concurrency
                for i in range(0, len(tasks), max_concurrent_ablations):
                    batch = tasks[i:i + max_concurrent_ablations]
                    await asyncio.gather(*batch)
            else:
                # Process all concurrently (unlimited)
                await asyncio.gather(*tasks)
            print()

    print(f"✓ Ablation complete for {input_file.name}!")


async def process_multiple_files_async(
    input_files: List[Path],
    output_dir: Path,
    strategy: str,
    think_mode: str,
    num_rollouts: int,
    parallel_files: int = 1,
    max_concurrent_ablations: int = 0
) -> None:
    """Process multiple files with configurable parallelism.

    Args:
        parallel_files: Number of files to process in parallel (default: 1 = sequential)
        max_concurrent_ablations: Max concurrent ablation tasks per file (0 = unlimited)
    """

    total_files = len(input_files)
    print(f"\n{'='*80}")
    print(f"Processing {total_files} files for ablation study")
    print(f"Parallel files: {parallel_files}")
    print(f"{'='*80}\n")

    if parallel_files == 1:
        # Sequential processing (original behavior)
        for idx, input_file in enumerate(input_files, 1):
            print(f"\n{'='*80}")
            print(f"File {idx}/{total_files}: {input_file.name}")
            print(f"{'='*80}")

            await process_ablation_async(
                input_file,
                output_dir,
                strategy,
                think_mode,
                num_rollouts,
                max_concurrent_ablations
            )
    else:
        # Parallel processing in batches
        for batch_start in range(0, total_files, parallel_files):
            batch_end = min(batch_start + parallel_files, total_files)
            batch = input_files[batch_start:batch_end]

            print(f"\n{'='*80}")
            print(f"Processing batch: files {batch_start+1}-{batch_end} of {total_files}")
            print(f"{'='*80}\n")

            tasks = [
                process_ablation_async(
                    input_file,
                    output_dir,
                    strategy,
                    think_mode,
                    num_rollouts,
                    max_concurrent_ablations
                )
                for input_file in batch
            ]
            await asyncio.gather(*tasks)

    print(f"\n{'='*80}")
    print(f"All {total_files} files processed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def get_input_files(input_path: str) -> List[Path]:
    """
    Get list of input files from:
    - Single file path
    - Directory (all .json files)
    - Glob pattern (e.g., "*.json")
    """
    files = []

    # Check if it's a glob pattern (contains * or ?)
    if '*' in input_path or '?' in input_path:
        matched_files = glob_module.glob(input_path)
        files = [Path(f) for f in matched_files if f.endswith('.json')]
    else:
        path = Path(input_path)

        if path.is_file():
            # Single file
            if path.suffix == '.json':
                files = [path]
            else:
                raise ValueError(f"File must be a JSON file: {path}")
        elif path.is_dir():
            # Directory - get all JSON files
            files = sorted(path.glob('*.json'))
        else:
            raise ValueError(f"Path not found: {path}")

    if not files:
        raise ValueError(f"No JSON files found at: {input_path}")

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Create paragraph ablation variants and generate rollouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/paragraph_ablation.py file.json --output-dir ablation_rollouts --num-rollouts 100

  # All files in directory
  python scripts/paragraph_ablation.py /path/to/data/ --output-dir ablation_rollouts --num-rollouts 100

  # Glob pattern
  python scripts/paragraph_ablation.py "/path/to/data/fic_*.json" --output-dir ablation_rollouts --num-rollouts 100
        """
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Input JSON file, directory, or glob pattern (e.g., "*.json")'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ablation_rollouts'),
        help='Output directory for ablation rollouts (default: ablation_rollouts/)'
    )
    parser.add_argument(
        '--strategy',
        choices=['keep_later', 'remove_after', 'random_sample', 'all'],
        default='all',
        help='Ablation strategy (default: all)'
    )
    parser.add_argument(
        '--think-mode',
        choices=['no_more_thinking', 'allow_more_thinking', 'both'],
        default='both',
        help='Think token mode (default: both)'
    )
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=100,
        help='Number of rollouts per ablation (default: 100)'
    )
    parser.add_argument(
        '--parallel-files',
        type=int,
        default=1,
        help='Number of files to process in parallel (default: 1 = sequential, 0 = all files at once)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=0,
        help='Max concurrent ablation tasks per file (default: 0 = unlimited, 5-10 recommended for API limits)'
    )

    args = parser.parse_args()

    # Get input files
    try:
        input_files = get_input_files(args.input_path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Found {len(input_files)} file(s) to process:")
    for f in input_files:
        print(f"  - {f}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine parallelism
    parallel_files = args.parallel_files
    if parallel_files == 0:
        parallel_files = len(input_files)  # Process all files at once

    # Run async processing
    asyncio.run(process_multiple_files_async(
        input_files,
        args.output_dir,
        args.strategy,
        args.think_mode,
        args.num_rollouts,
        parallel_files,
        args.max_workers
    ))

    return 0


if __name__ == "__main__":
    exit(main())
