#!/usr/bin/env python3
"""
Generate stories for twist ablation study using rollout library.
Uses twist prompts with combinations of NAME, GOAL, and TWIST_PHRASE.
"""

import yaml
import json
import argparse
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import rollouts
sys.path.insert(0, str(Path(__file__).parent.parent))
from rollouts import RolloutsClient
from twist.config import (
    TWIST_PROMPTS_YAML,
    OUTPUT_DIR_TWIST,
    MODEL_NAME,
    CACHE_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_N_SAMPLES
)

# Load environment variables from .env file
load_dotenv()


def load_prompts(yaml_path: Path) -> dict:
    """Load prompts from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_sample_to_file(result, combined_prompt: str, prompt_data: dict, output_dir: Path):
    """Save generated sample to JSON file with metadata."""
    # Use prompt ID for filename
    prompt_id = prompt_data.get('id', 'unknown')

    # Save to file as JSON
    output_file = output_dir / f"{prompt_id}.json"
    output_data = {
        "prompt_id": prompt_id,
        "protagonist": prompt_data.get('protagonist'),
        "goal": prompt_data.get('goal'),
        "twist_phrase": prompt_data.get('twist_phrase'),
        "prompt_text": prompt_data.get('text'),
        "prompt_metadata": prompt_data.get('metadata', {}),
        "combined_prompt": combined_prompt,
        "response": result.to_dict() if hasattr(result, 'to_dict') else result,
        "model": MODEL_NAME,
        "generation_config": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "n_samples": DEFAULT_N_SAMPLES
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")


def create_client():
    """Create and return a RolloutsClient instance."""
    return RolloutsClient(
        model=MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        cache_dir=str(CACHE_PATH)
    )


async def generate_sample_async(client: RolloutsClient, combined_prompt: str, prompt_data: dict, output_dir: Path):
    """Generate a single sample using the rollouts library (async)."""
    prompt_id = prompt_data.get('id', 'unknown')
    print(f"Generating {prompt_id}...")

    response = await client.agenerate(prompt=combined_prompt, n_samples=DEFAULT_N_SAMPLES)

    # Validate that we got exactly one response
    if not hasattr(response, 'responses') or len(response.responses) != DEFAULT_N_SAMPLES:
        num_responses = len(response.responses) if hasattr(response, 'responses') else 0
        raise ValueError(
            f"Expected {DEFAULT_N_SAMPLES} response(s), but got {num_responses} responses for {prompt_id}"
        )

    save_sample_to_file(response, combined_prompt, prompt_data, output_dir)


def generate_sample_sync(client: RolloutsClient, combined_prompt: str, prompt_data: dict, output_dir: Path):
    """Generate a single sample using the rollouts library (sync)."""
    prompt_id = prompt_data.get('id', 'unknown')
    print(f"Generating {prompt_id}...")

    response = client.generate(prompt=combined_prompt, n_samples=DEFAULT_N_SAMPLES)

    # Validate that we got exactly one response
    if not hasattr(response, 'responses') or len(response.responses) != DEFAULT_N_SAMPLES:
        num_responses = len(response.responses) if hasattr(response, 'responses') else 0
        raise ValueError(
            f"Expected {DEFAULT_N_SAMPLES} response(s), but got {num_responses} responses for {prompt_id}"
        )

    save_sample_to_file(response, combined_prompt, prompt_data, output_dir)


async def main_async(args):
    """Async version of main for concurrent generation."""
    client = create_client()
    data = load_prompts(TWIST_PROMPTS_YAML)

    # Create output directory
    output_dir = OUTPUT_DIR_TWIST
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get system prompt and prompts list
    system_prompt = data.get('system_prompt', '')
    prompts_list = data['prompts']

    # Filter to specific prompt range if specified
    if args.start_id or args.end_id:
        start_idx = int(args.start_id.split('_')[1]) if args.start_id else 1
        end_idx = int(args.end_id.split('_')[1]) if args.end_id else len(prompts_list)
        prompts_list = [p for p in prompts_list
                       if start_idx <= int(p['id'].split('_')[1]) <= end_idx]
        print(f"Filtering to prompts {start_idx:03d} to {end_idx:03d}")

    # Filter to single prompt if specified
    if args.single_id:
        prompts_list = [p for p in prompts_list if p.get('id') == args.single_id]
        if not prompts_list:
            print(f"Error: No prompt found with id '{args.single_id}'")
            return
        print(f"Testing with single prompt: {args.single_id}")

    # Generate samples concurrently
    tasks = []
    for prompt_data in prompts_list:
        prompt_text = prompt_data.get('text', '')
        combined_prompt = f"{system_prompt}\n\n{prompt_text}"
        tasks.append(generate_sample_async(client, combined_prompt, prompt_data, output_dir))

    print(f"\nGenerating {len(tasks)} samples concurrently...")
    await asyncio.gather(*tasks)

    print(f"\n{'='*80}")
    print(f"All {len(tasks)} twist stories generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def main_sync(args):
    """Sync version of main for sequential generation."""
    client = create_client()
    data = load_prompts(TWIST_PROMPTS_YAML)

    # Create output directory
    output_dir = OUTPUT_DIR_TWIST
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get system prompt and prompts list
    system_prompt = data.get('system_prompt', '')
    prompts_list = data['prompts']

    # Filter to specific prompt range if specified
    if args.start_id or args.end_id:
        start_idx = int(args.start_id.split('_')[1]) if args.start_id else 1
        end_idx = int(args.end_id.split('_')[1]) if args.end_id else len(prompts_list)
        prompts_list = [p for p in prompts_list
                       if start_idx <= int(p['id'].split('_')[1]) <= end_idx]
        print(f"Filtering to prompts {start_idx:03d} to {end_idx:03d}\n")

    # Filter to single prompt if specified
    if args.single_id:
        prompts_list = [p for p in prompts_list if p.get('id') == args.single_id]
        if not prompts_list:
            print(f"Error: No prompt found with id '{args.single_id}'")
            return
        print(f"Testing with single prompt: {args.single_id}\n")

    total_samples = len(prompts_list)
    current = 0

    # Generate samples sequentially
    for prompt_data in prompts_list:
        prompt_text = prompt_data.get('text', '')
        combined_prompt = f"{system_prompt}\n\n{prompt_text}"
        generate_sample_sync(client, combined_prompt, prompt_data, output_dir)

        current += 1
        print(f"Progress: {current}/{total_samples} samples completed\n")

    print(f"\n{'='*80}")
    print(f"All {total_samples} twist stories generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate twist ablation stories using combinations of NAME, GOAL, and TWIST_PHRASE'
    )
    parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        help='Use async mode for concurrent generation (faster)'
    )
    parser.add_argument(
        '--single-id',
        type=str,
        metavar='PROMPT_ID',
        help='Test with only a single prompt (e.g., twist_001)'
    )
    parser.add_argument(
        '--start-id',
        type=str,
        metavar='START_ID',
        help='Start from this prompt ID (e.g., twist_001)'
    )
    parser.add_argument(
        '--end-id',
        type=str,
        metavar='END_ID',
        help='End at this prompt ID (e.g., twist_125)'
    )

    args = parser.parse_args()

    if args.use_async:
        asyncio.run(main_async(args))
    else:
        main_sync(args)


if __name__ == "__main__":
    main()
