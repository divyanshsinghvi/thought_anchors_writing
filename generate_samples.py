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
from config import PROMPTS_YAML, FICTIONAL_PROMPTS_YAML, OUTPUT_DIR_NON_FICTION, OUTPUT_DIR_FICTION, MODEL_NAME, CACHE_PATH

# Load environment variables from .env file
load_dotenv()


def load_prompts(yaml_path: str) -> dict:
    """Load prompts from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_sample_to_file(result, combined_prompt: str, prompt_data: dict, output_dir: Path, sys_prompt_name: str):
    """Save generated sample to JSON file with metadata."""
    # Use prompt ID for filename
    prompt_id = prompt_data.get('id', f"sample_{prompt_data.get('category', 'unknown')}")
    
    # Save to file as JSON
    output_file = output_dir / f"{prompt_id}_{sys_prompt_name}.json"
    output_data = {
        "prompt_id": prompt_data.get('id'),
        "prompt_text": prompt_data.get('text'),
        "prompt_metadata": {
            "category": prompt_data.get('category'),
            "subcategory": prompt_data.get('subcategory'),
            "difficulty": prompt_data.get('difficulty'),
            "expected_length": prompt_data.get('expected_length'),
            "factual_density": prompt_data.get('factual_density'),
            "narrative_style": prompt_data.get('narrative_style'),
            "expected_reasoning_focus": prompt_data.get('expected_reasoning_focus', []),
            "expected_story_elements": prompt_data.get('expected_story_elements', [])
        },
        "system_prompt_name": sys_prompt_name,
        "combined_prompt": combined_prompt,
        "response": result.to_dict() if hasattr(result, 'to_dict') else result,
        "model": MODEL_NAME,
        "generation_config": {
            "temperature": 0.6,
            "top_p": 0.95
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")


def get_paths_and_dirs(args):
    """Get input file and output base directory based on type."""
    if args.type == 'fiction':
        prompts_file = FICTIONAL_PROMPTS_YAML
        output_base_dir = OUTPUT_DIR_FICTION
    else:  # non-fiction
        prompts_file = PROMPTS_YAML
        output_base_dir = OUTPUT_DIR_NON_FICTION

    return prompts_file, output_base_dir


def get_system_prompt_configs(data, output_base_dir: Path):
    """Extract system prompts and create output directories dynamically."""
    sys_prompt_configs = []

    for sys_prompt_name, sys_prompt_text in data['system_prompts'].items():
        output_dir = output_base_dir / sys_prompt_name
        sys_prompt_configs.append((sys_prompt_text, output_dir, sys_prompt_name))

    # Create all output directories
    for _, output_dir, _ in sys_prompt_configs:
        output_dir.mkdir(parents=True, exist_ok=True)

    return sys_prompt_configs


def create_client():
    """Create and return a RolloutsClient instance."""
    return RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
        cache_dir=str(CACHE_PATH)
    )


async def generate_sample_async(client: RolloutsClient, combined_prompt: str, prompt_data: dict, output_dir: Path, sys_prompt_name: str):
    """Generate a single sample using the rollouts library (async)."""
    prompt_id = prompt_data.get('id', 'unknown')
    print(f"Generating {prompt_id} with {sys_prompt_name}...")

    response = await client.agenerate(prompt=combined_prompt, n_samples=1)

    # Validate that we got exactly one response
    if not hasattr(response, 'responses') or len(response.responses) != 1:
        num_responses = len(response.responses) if hasattr(response, 'responses') else 0
        raise ValueError(
            f"Expected single response (n_samples=1), but got {num_responses} responses for {prompt_id}"
        )

    save_sample_to_file(response, combined_prompt, prompt_data, output_dir, sys_prompt_name)


def generate_sample_sync(client: RolloutsClient, combined_prompt: str, prompt_data: dict, output_dir: Path, sys_prompt_name: str):
    """Generate a single sample using the rollouts library (sync)."""
    prompt_id = prompt_data.get('id', 'unknown')
    print(f"Generating {prompt_id} with {sys_prompt_name}...")

    response = client.generate(prompt=combined_prompt, n_samples=1)

    # Validate that we got exactly one response
    if not hasattr(response, 'responses') or len(response.responses) != 1:
        num_responses = len(response.responses) if hasattr(response, 'responses') else 0
        raise ValueError(
            f"Expected single response (n_samples=1), but got {num_responses} responses for {prompt_id}"
        )

    save_sample_to_file(response, combined_prompt, prompt_data, output_dir, sys_prompt_name)


async def main_async(args):
    """Async version of main for concurrent generation."""
    prompts_file, output_base_dir = get_paths_and_dirs(args)
    client = create_client()
    data = load_prompts(str(prompts_file))

    # Get system prompt configurations dynamically
    sys_prompt_configs = get_system_prompt_configs(data, output_base_dir)

    # Generate samples for each prompt with all system prompts concurrently
    tasks = []
    prompts_list = data['prompts']

    # Filter to single prompt if specified
    if args.single_file:
        prompts_list = [p for p in prompts_list if (p.get('id') if isinstance(p, dict) else f"prompt_{prompts_list.index(p) + 1:03d}") == args.single_file]
        if not prompts_list:
            print(f"Error: No prompt found with id '{args.single_file}'")
            return
        print(f"Testing with single prompt: {args.single_file}")

    for sys_prompt, output_dir, sys_prompt_name in sys_prompt_configs:
        print(f"Queueing samples with {sys_prompt_name}")
        for prompt_data in prompts_list:
            # Handle both string prompts (old format) and dict prompts (new format)
            if isinstance(prompt_data, str):
                prompt_data = {
                    'text': prompt_data,
                    'id': f"prompt_{prompts_list.index(prompt_data) + 1:03d}"
                }
            
            prompt_text = prompt_data.get('text', prompt_data)
            combined_prompt = f"{sys_prompt}\n\n{prompt_text}"
            tasks.append(generate_sample_async(client, combined_prompt, prompt_data, output_dir, sys_prompt_name))

    print(f"\nGenerating {len(tasks)} samples concurrently...")
    await asyncio.gather(*tasks)

    print(f"\n{'='*80}")
    print(f"All samples generated successfully for {args.type}!")
    print(f"Output directories:")
    for config in sys_prompt_configs:
        print(f"  - {config[1]}")
    print(f"{'='*80}\n")


def main_sync(args):
    """Sync version of main for sequential generation."""
    prompts_file, output_base_dir = get_paths_and_dirs(args)
    client = create_client()
    data = load_prompts(str(prompts_file))

    # Get system prompt configurations dynamically
    sys_prompt_configs = get_system_prompt_configs(data, output_base_dir)

    prompts_list = data['prompts']

    # Filter to single prompt if specified
    if args.single_file:
        prompts_list = [p for p in prompts_list if (p.get('id') if isinstance(p, dict) else f"prompt_{prompts_list.index(p) + 1:03d}") == args.single_file]
        if not prompts_list:
            print(f"Error: No prompt found with id '{args.single_file}'")
            return
        print(f"Testing with single prompt: {args.single_file}\n")

    total_samples = len(prompts_list) * len(sys_prompt_configs)
    current = 0

    # Generate samples for each prompt with all system prompts sequentially
    for sys_prompt, output_dir, sys_prompt_name in sys_prompt_configs:
        print(f"\n{'='*80}\nGenerating samples with {sys_prompt_name}\n{'='*80}")
        for prompt_data in prompts_list:
            # Handle both string prompts (old format) and dict prompts (new format)
            if isinstance(prompt_data, str):
                prompt_data = {
                    'text': prompt_data,
                    'id': f"prompt_{prompts_list.index(prompt_data) + 1:03d}"
                }
            
            prompt_text = prompt_data.get('text', prompt_data)
            combined_prompt = f"{sys_prompt}\n\n{prompt_text}"
            generate_sample_sync(client, combined_prompt, prompt_data, output_dir, sys_prompt_name)
            
            current += 1
            print(f"Progress: {current}/{total_samples} samples completed\n")

    print(f"\n{'='*80}")
    print(f"All samples generated successfully for {args.type}!")
    print(f"Output directories:")
    for config in sys_prompt_configs:
        print(f"  - {config[1]}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate writing samples for fiction or non-fiction prompts')
    parser.add_argument('--type', type=str, choices=['fiction', 'non-fiction'], required=True,
                        help='Type of prompts to process: fiction or non-fiction')
    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Use async mode for concurrent generation (faster)')
    parser.add_argument('--single-file', type=str, metavar='PROMPT_ID',
                        help='Test with only a single prompt file (e.g., prompt_001)')
    args = parser.parse_args()

    if args.use_async:
        asyncio.run(main_async(args))
    else:
        main_sync(args)


if __name__ == "__main__":
    main()