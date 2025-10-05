#!/usr/bin/env python3
"""
Twist Ablation Study for Chain-of-Thought Reasoning

This script tests how changing the twist affects story generation while keeping
the original reasoning fixed.

Strategy:
1. Take reasoning from baseline story (e.g., twist_001)
2. Apply it to prompts with different twists (e.g., twist_002, twist_003)
3. Test two think modes:
   - no_more_thinking: Close </think> tag - force use of baseline reasoning only
   - allow_more_thinking: Leave <think> tag open - allow model to add more reasoning

Usage:
    # Automatic mode: test baseline against all other twists with same protagonist+goal
    python twist/twist_ablation.py \
        --baseline twist_001 \
        --auto-find-twists \
        --num-rollouts 10 \
        --think-mode both

    # Manual mode: specify exact twists to test
    python twist/twist_ablation.py \
        --baseline twist_001 \
        --ablate-twists twist_002 twist_003 \
        --num-rollouts 10 \
        --think-mode both
"""

import json
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts import RolloutsClient
from twist.config import MODEL_NAME, CACHE_PATH, OUTPUT_DIR_BASE_PATH

# Load environment variables
load_dotenv()


def load_baseline_story(baseline_id: str, stories_dir: Path) -> Dict:
    """Load the baseline story JSON file."""
    baseline_file = stories_dir / f"{baseline_id}.json"

    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline story not found: {baseline_file}")

    with open(baseline_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_twist_prompt(twist_id: str, stories_dir: Path) -> Dict:
    """Load a twist story to get the prompt with different twist."""
    twist_file = stories_dir / f"{twist_id}.json"

    if not twist_file.exists():
        raise FileNotFoundError(f"Twist story not found: {twist_file}")

    with open(twist_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_matching_twists(baseline_data: Dict, stories_dir: Path) -> List[str]:
    """
    Find all twist stories with same protagonist+goal but different twist.

    Returns list of twist IDs (e.g., ['twist_002', 'twist_003', ...])
    """
    baseline_protagonist = baseline_data.get('protagonist')
    baseline_goal = baseline_data.get('goal')
    baseline_twist = baseline_data.get('twist_phrase')
    baseline_id = baseline_data.get('prompt_id')

    matching_twists = []

    # Iterate through all story files
    for story_file in sorted(stories_dir.glob('twist_*.json')):
        twist_id = story_file.stem

        # Skip the baseline itself
        if twist_id == baseline_id:
            continue

        # Load and check if protagonist+goal match
        with open(story_file, 'r', encoding='utf-8') as f:
            twist_data = json.load(f)

        if (twist_data.get('protagonist') == baseline_protagonist and
            twist_data.get('goal') == baseline_goal and
            twist_data.get('twist_phrase') != baseline_twist):
            matching_twists.append(twist_id)

    return matching_twists


def create_ablated_prompt(
    baseline_reasoning: str,
    twist_prompt_text: str,
    system_prompt: str,
    think_mode: str = 'no_more_thinking'
) -> str:
    """
    Create a new prompt with baseline reasoning + different twist prompt.

    Think modes:
    - no_more_thinking: Close <think> tag after baseline reasoning
      [System prompt]\n[New twist prompt]\n<think>\n[baseline reasoning]\n</think>

    - allow_more_thinking: Keep <think> tag open for model to continue thinking
      [System prompt]\n[New twist prompt]\n<think>\n[baseline reasoning]
    """
    # Extract just the system prompt (before the actual writing prompt)
    # The combined_prompt format is: "system_prompt\n\nwriting_prompt"
    # We need to replace the writing_prompt part with twist_prompt_text

    if think_mode == 'no_more_thinking':
        ablated_prompt = f"{system_prompt}\n\n{twist_prompt_text}\n\n<think>\n{baseline_reasoning}\n</think>"
    else:  # allow_more_thinking
        ablated_prompt = f"{system_prompt}\n\n{twist_prompt_text}\n\n<think>\n{baseline_reasoning}\n"

    return ablated_prompt


async def generate_ablation_rollouts(
    client: RolloutsClient,
    ablated_prompt: str,
    baseline_data: Dict,
    twist_data: Dict,
    num_rollouts: int,
    output_dir: Path,
    think_mode: str
) -> None:
    """Generate rollouts for a single twist ablation and save each rollout separately."""

    baseline_id = baseline_data['prompt_id']
    twist_id = twist_data['prompt_id']

    print(f"  Generating {num_rollouts} rollouts for {twist_id} with reasoning from {baseline_id} (think: {think_mode})...")

    # Generate rollouts
    response = await client.agenerate(prompt=ablated_prompt, n_samples=num_rollouts)

    # Extract baseline reasoning for metadata
    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']

    # Create base metadata that's common to all rollouts
    base_metadata = {
        'baseline_prompt_id': baseline_id,
        'twist_prompt_id': twist_id,
        'baseline_protagonist': baseline_data.get('protagonist'),
        'baseline_goal': baseline_data.get('goal'),
        'baseline_twist': baseline_data.get('twist_phrase'),
        'new_twist': twist_data.get('twist_phrase'),
        'prompt_text': twist_data['prompt_text'],
        'ablation_metadata': {
            'think_mode': think_mode,
            'baseline_file': f"{baseline_id}.json",
            'uses_baseline_reasoning': True
        },
        'baseline_reasoning': baseline_reasoning,
        'ablated_prompt': ablated_prompt,
        'baseline_content': baseline_data['response']['responses'][0]['content'],
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

        # Create output file: twist_NNN/think_mode/rollout_NNN.json
        rollout_file = output_dir / f"rollout_{rollout_idx:03d}.json"
        with open(rollout_file, 'w', encoding='utf-8') as f:
            json.dump(rollout_output, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Saved {len(rollouts)} rollouts to {output_dir}")


async def run_twist_ablation(
    baseline_id: str,
    twist_ids: List[str],
    num_rollouts: int,
    think_modes: List[str],
    stories_dir: Path,
    output_base_dir: Path,
    auto_find: bool = False
):
    """Run twist ablation study for specified baseline and twist variations."""

    # Load baseline story
    print(f"Loading baseline story: {baseline_id}")
    baseline_data = load_baseline_story(baseline_id, stories_dir)
    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']

    # Auto-find matching twists if requested
    if auto_find:
        print(f"Auto-finding twists with same protagonist+goal...")
        twist_ids = find_matching_twists(baseline_data, stories_dir)
        print(f"Found {len(twist_ids)} matching twists: {twist_ids}")

    if not twist_ids:
        print("No twist IDs to process. Exiting.")
        return

    # Extract system prompt from combined_prompt
    combined_prompt = baseline_data['combined_prompt']
    # System prompt is everything before the actual writing prompt
    # Format: "system_prompt\n\nYou will first think through..."
    parts = combined_prompt.split('\n\n', 1)
    system_prompt = parts[0]

    print(f"\nBaseline configuration:")
    print(f"  Protagonist: {baseline_data.get('protagonist')}")
    print(f"  Goal: {baseline_data.get('goal')}")
    print(f"  Baseline twist: {baseline_data.get('twist_phrase')}")
    print(f"  Reasoning length: {len(baseline_reasoning)} chars")
    print(f"\nRunning ablation for {len(twist_ids)} twist(s) with {len(think_modes)} think mode(s)")

    # Create client
    client = RolloutsClient(
        model=MODEL_NAME,
        temperature=0.6,
        top_p=0.95,
        cache_dir=str(CACHE_PATH)
    )

    # Process each twist
    for twist_id in twist_ids:
        print(f"\n{'='*80}")
        print(f"Processing {twist_id}")
        print(f"{'='*80}")

        # Load twist story to get the prompt with different twist
        twist_data = load_twist_prompt(twist_id, stories_dir)
        print(f"New twist: {twist_data.get('twist_phrase')}")

        # Get the writing prompt text (without system prompt)
        twist_prompt_text = twist_data['prompt_text']

        # Generate rollouts for each think mode
        for think_mode in think_modes:
            print(f"\n  Think mode: {think_mode}")

            # Create ablated prompt
            ablated_prompt = create_ablated_prompt(
                baseline_reasoning=baseline_reasoning,
                twist_prompt_text=twist_prompt_text,
                system_prompt=system_prompt,
                think_mode=think_mode
            )

            # Print the ablated prompt for analysis
            print(f"\n  {'─'*76}")
            print(f"  ABLATED PROMPT ({think_mode}):")
            print(f"  {'─'*76}")
            print(ablated_prompt[:500] + "..." if len(ablated_prompt) > 500 else ablated_prompt)
            print(f"  {'─'*76}")
            print(f"  Total prompt length: {len(ablated_prompt)} chars")
            print(f"  Baseline reasoning length: {len(baseline_reasoning)} chars")
            print(f"  {'─'*76}\n")

            # Create output directory: ablation_outputs/twist_NNN/think_mode/
            output_dir = output_base_dir / twist_id / think_mode
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate rollouts
            await generate_ablation_rollouts(
                client=client,
                ablated_prompt=ablated_prompt,
                baseline_data=baseline_data,
                twist_data=twist_data,
                num_rollouts=num_rollouts,
                output_dir=output_dir,
                think_mode=think_mode
            )

    print(f"\n{'='*80}")
    print(f"Twist ablation complete!")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run twist ablation study: test how changing twist affects generation with fixed reasoning'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline story ID to extract reasoning from (e.g., twist_001)'
    )
    parser.add_argument(
        '--ablate-twists',
        nargs='+',
        help='Twist story IDs to apply baseline reasoning to (e.g., twist_002 twist_003)'
    )
    parser.add_argument(
        '--auto-find-twists',
        action='store_true',
        help='Automatically find all twists with same protagonist+goal as baseline'
    )
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=10,
        help='Number of rollouts to generate per configuration (default: 10)'
    )
    parser.add_argument(
        '--think-mode',
        type=str,
        choices=['no_more_thinking', 'allow_more_thinking', 'both'],
        default='both',
        help='Think token mode (default: both)'
    )
    parser.add_argument(
        '--stories-dir',
        type=Path,
        default=Path('/mnt/d/code/open_source/mats/thought_anchors_writing/data/twist/stories'),
        help='Directory containing baseline stories'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR_BASE_PATH / 'ablation_outputs',
        help='Base output directory for ablation results'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.auto_find_twists and not args.ablate_twists:
        parser.error("Either --auto-find-twists or --ablate-twists must be specified")

    # Determine think modes to run
    if args.think_mode == 'both':
        think_modes = ['no_more_thinking', 'allow_more_thinking']
    else:
        think_modes = [args.think_mode]

    # Run ablation
    asyncio.run(run_twist_ablation(
        baseline_id=args.baseline,
        twist_ids=args.ablate_twists if args.ablate_twists else [],
        num_rollouts=args.num_rollouts,
        think_modes=think_modes,
        stories_dir=args.stories_dir,
        output_base_dir=args.output_dir,
        auto_find=args.auto_find_twists
    ))


if __name__ == "__main__":
    main()
