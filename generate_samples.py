#!/usr/bin/env python3
"""
Generate samples for non-fictional writing prompts using rollout library.
Uses qwen3-14b model with think tags for improved reasoning.
"""

import yaml
import json
from dotenv import load_dotenv
from rollouts import RolloutsClient
from config import PROMPTS_YAML, OUTPUT_DIR, MODEL_NAME

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
    # Initialize rollout with a smaller, faster model for testing
    client = RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
    )

    # Load prompts
    data = load_prompts(str(PROMPTS_YAML))

    sys_prompt_1 = data['sys_prompt_1']
    sys_prompt_2 = data['sys_prompt_2']
    prompts = data['prompts']

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate one sample for each prompt using sys_prompt_1
    for idx, prompt in enumerate(prompts, 1):
        # Extract a short name from the prompt for the filename
        prompt_name = prompt.split(']')[1].strip()[:60].replace(' ', '_').replace(',', '')
        

        combined_prompt = f"{sys_prompt_1}\n\n{prompt}"
        # Generate sample
        response = generate_sample(client, combined_prompt, prompt_name)

        # Save to file as JSON
        output_file = OUTPUT_DIR / f"sample_{idx:02d}_{prompt_name}.json"
        output_data = {
            "prompt": combined_prompt,
            "response": response.to_dict() if hasattr(response, 'to_dict') else response,
            "model": MODEL_NAME,
            "sample_id": idx,
            "prompt_name": prompt_name
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"All samples generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
