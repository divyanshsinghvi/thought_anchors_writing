#!/usr/bin/env python3
"""
Generate samples for fiction/non-fiction writing prompts using rollout library.
Uses qwen model for generation.
"""

import yaml
import json
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from rollouts import RolloutsClient
from config import PROMPTS_YAML, OUTPUT_DIR_NON_FICTION, OUTPUT_DIR_FICTION, MODEL_NAME, BASE_DIR, CACHE_PATH

# Load environment variables from .env file
load_dotenv()


def load_prompts(yaml_path: str) -> dict:
    """Load prompts from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_sample_to_file(result, combined_prompt: str, idx: int, prompt_name: str, output_dir: Path, sys_prompt_name: str):
    """Save generated sample to JSON file."""
    # Extract short name for filename
    prompt_name_short = prompt_name.split(']')[1].strip()[:60].replace(' ', '_').replace(',', '') if ']' in prompt_name else prompt_name[:60].replace(' ', '_').replace(',', '')

    # Save to file as JSON
    output_file = output_dir / f"sample_{idx:02d}_{prompt_name_short}.json"
    output_data = {
        "prompt": combined_prompt,
        "response": result.to_dict() if hasattr(result, 'to_dict') else result,
        "model": MODEL_NAME,
        "system_prompt": sys_prompt_name,
        "sample_id": idx,
        "prompt_name": prompt_name_short
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")


def get_paths_and_dirs(args):
    """Get input file and output directories based on type."""
    if args.type == 'fiction':
        prompts_file = BASE_DIR / "fictional_prompts.yaml"
        output_base_dir = OUTPUT_DIR_FICTION
    else:  # non-fiction
        prompts_file = PROMPTS_YAML
        output_base_dir = OUTPUT_DIR_NON_FICTION

    sys_prompt_1_dir = output_base_dir / "sys_prompt_1"
    sys_prompt_2_dir = output_base_dir / "sys_prompt_2"
    sys_prompt_1_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt_2_dir.mkdir(parents=True, exist_ok=True)

    return prompts_file, sys_prompt_1_dir, sys_prompt_2_dir


def create_client():
    """Create and return a RolloutsClient instance."""
    return RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
        cache_dir=str(CACHE_PATH)
    )


async def generate_sample_async(client: RolloutsClient, combined_prompt: str, prompt_name: str, idx: int, output_dir: Path, sys_prompt_name: str):
    """Generate a single sample using the rollouts library (async)."""
    print(f"Generating sample for: {prompt_name}")
    response = await client.agenerate(prompt=combined_prompt, n_samples=1)
    result = response[0] if isinstance(response, list) else response
    save_sample_to_file(result, combined_prompt, idx, prompt_name, output_dir, sys_prompt_name)


def generate_sample_sync(client: RolloutsClient, combined_prompt: str, prompt_name: str, idx: int, output_dir: Path, sys_prompt_name: str):
    """Generate a single sample using the rollouts library (sync)."""
    print(f"Generating sample for: {prompt_name}")
    response = client.generate(prompt=combined_prompt, n_samples=1)
    result = response[0] if isinstance(response, list) else response
    save_sample_to_file(result, combined_prompt, idx, prompt_name, output_dir, sys_prompt_name)


async def main_async(args):
    """Async version of main for concurrent generation."""
    prompts_file, sys_prompt_1_dir, sys_prompt_2_dir = get_paths_and_dirs(args)
    client = create_client()
    data = load_prompts(str(prompts_file))

    sys_prompt_configs = [
        (data['sys_prompt_1'], sys_prompt_1_dir, "sys_prompt_1"),
        (data['sys_prompt_2'], sys_prompt_2_dir, "sys_prompt_2")
    ]

    # Generate samples for each prompt with both system prompts concurrently
    tasks = []
    for sys_prompt, output_dir, sys_prompt_name in sys_prompt_configs:
        print(f"Queueing samples with {sys_prompt_name}")
        for idx, prompt in enumerate(data['prompts'], 1):
            combined_prompt = f"{sys_prompt}\n\n{prompt}"
            tasks.append(generate_sample_async(client, combined_prompt, prompt, idx, output_dir, sys_prompt_name))

    print(f"\nGenerating {len(tasks)} samples concurrently...")
    await asyncio.gather(*tasks)

    print(f"\n{'='*80}")
    print(f"All samples generated successfully for {args.type}!")
    print(f"Output directories:\n  - {sys_prompt_1_dir}\n  - {sys_prompt_2_dir}")
    print(f"{'='*80}\n")


def main_sync(args):
    """Sync version of main for sequential generation."""
    prompts_file, sys_prompt_1_dir, sys_prompt_2_dir = get_paths_and_dirs(args)
    client = create_client()
    data = load_prompts(str(prompts_file))

    sys_prompt_configs = [
        (data['sys_prompt_1'], sys_prompt_1_dir, "sys_prompt_1"),
        (data['sys_prompt_2'], sys_prompt_2_dir, "sys_prompt_2")
    ]

    # Generate samples for each prompt with both system prompts sequentially
    for sys_prompt, output_dir, sys_prompt_name in sys_prompt_configs:
        print(f"\n{'='*80}\nGenerating samples with {sys_prompt_name}\n{'='*80}")
        for idx, prompt in enumerate(data['prompts'], 1):
            combined_prompt = f"{sys_prompt}\n\n{prompt}"
            generate_sample_sync(client, combined_prompt, prompt, idx, output_dir, sys_prompt_name)

    print(f"\n{'='*80}")
    print(f"All samples generated successfully for {args.type}!")
    print(f"Output directories:\n  - {sys_prompt_1_dir}\n  - {sys_prompt_2_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate writing samples for fiction or non-fiction prompts')
    parser.add_argument('--type', type=str, choices=['fiction', 'non-fiction'], required=True,
                        help='Type of prompts to process: fiction or non-fiction')
    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use async mode for concurrent generation (faster)')
    args = parser.parse_args()

    if args.use_async:
        asyncio.run(main_async(args))
    else:
        main_sync(args)


if __name__ == "__main__":
    main()
