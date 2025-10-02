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


def generate_sample(client: RolloutsClient, combined_prompt: str, prompt_name: str) -> str:
    """Generate a single sample using the rollouts library."""
    # Combine system and user prompts into a single prompt

    print(f"\n{'='*80}")
    print(f"Generating sample for: {prompt_name}")
    print(f"{'='*80}\n")

    response = client.generate(prompt=combined_prompt, n_samples=1)
    return response


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate writing samples for fiction or non-fiction prompts')
    parser.add_argument('--type', type=str, choices=['fiction', 'non-fiction'], required=True,
                        help='Type of prompts to process: fiction or non-fiction')
    args = parser.parse_args()

    # Determine input file and output directory based on type
    if args.type == 'fiction':
        prompts_file = BASE_DIR / "fictional_prompts.yaml"
        output_base_dir = OUTPUT_DIR_FICTION
    else:  # non-fiction
        prompts_file = PROMPTS_YAML  # non_fictional_prompts.yaml
        output_base_dir = OUTPUT_DIR_NON_FICTION

    # Initialize rollout with cache directory
    client = RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
        cache_dir=str(CACHE_PATH)
    )

    # Load prompts
    data = load_prompts(str(prompts_file))

    sys_prompt_1 = data['sys_prompt_1']
    sys_prompt_2 = data['sys_prompt_2']
    prompts = data['prompts']

    # Create output directories
    sys_prompt_1_dir = output_base_dir / "sys_prompt_1"
    sys_prompt_2_dir = output_base_dir / "sys_prompt_2"
    sys_prompt_1_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt_2_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples for each prompt with both system prompts
    for sys_prompt, output_dir, sys_prompt_name in [
        (sys_prompt_1, sys_prompt_1_dir, "sys_prompt_1"),
        (sys_prompt_2, sys_prompt_2_dir, "sys_prompt_2")
    ]:
        print(f"\n{'='*80}")
        print(f"Generating samples with {sys_prompt_name}")
        print(f"{'='*80}\n")

        for idx, prompt in enumerate(prompts, 1):
            # Extract a short name from the prompt for the filename
            prompt_name = prompt.split(']')[1].strip()[:60].replace(' ', '_').replace(',', '')

            combined_prompt = f"{sys_prompt}\n\n{prompt}"
            # Generate sample
            response = generate_sample(client, combined_prompt, prompt_name)

            # Save to file as JSON
            output_file = output_dir / f"sample_{idx:02d}_{prompt_name}.json"
            output_data = {
                "prompt": combined_prompt,
                "response": response.to_dict() if hasattr(response, 'to_dict') else response,
                "model": MODEL_NAME,
                "system_prompt": sys_prompt_name,
                "sample_id": idx,
                "prompt_name": prompt_name
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"✓ Saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"All samples generated successfully for {args.type}!")
    print(f"Output directories:")
    print(f"  - {sys_prompt_1_dir}")
    print(f"  - {sys_prompt_2_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
