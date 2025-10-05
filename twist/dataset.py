#!/usr/bin/env python3
"""
Generate twist ablation dataset by creating all combinations of NAME, GOAL, and TWIST_PHRASE.
Saves prompts to prompts.yaml for later generation.
"""

import yaml
import argparse
from pathlib import Path
from itertools import product
from typing import List, Dict
from config import TWIST_PROMPTS_YAML

# Prompt template
PROMPT_TEMPLATE = """You will first think through a PLAN THOROUGHLY, then write a short story (70–120 tokens).

Outline:
Protagonist: {name}
Goal: {goal}
Twist: {twist_phrase}

Guidelines:
* Make sure the Protagonist, Goal and Twist are an integral part of the story.
* Do not include meta-comments about writing; just output the story itself."""


# PROMPT_TEMPLATE = """You will first think through a PLAN THOROUGHLY, then write a short story (around 3-5 sentences).

# Outline:
# Protagonist: {name}
# Goal: {goal}
# Twist: {twist_phrase}

# Guidelines:
# * Do not include meta-comments about writing; just output the story itself."""



def generate_dataset(
    names: List[str],
    goals: List[str],
    twist_phrases: List[str],
    output_file: Path = TWIST_PROMPTS_YAML
) -> Dict:
    """
    Generate all combinations of names, goals, and twist phrases.

    Args:
        names: List of protagonist names
        goals: List of protagonist goals
        twist_phrases: List of narrative twists
        output_file: Path to save the generated YAML dataset

    Returns:
        Dictionary containing the dataset with metadata
    """

    # Generate all combinations
    combinations = list(product(names, goals, twist_phrases))

    # Create prompt entries
    prompts = []
    for idx, (name, goal, twist) in enumerate(combinations, start=1):
        prompt_id = f"twist_{idx:03d}"

        prompt_entry = {
            "id": prompt_id,
            "protagonist": name,
            "goal": goal,
            "twist_phrase": twist,
            "text": PROMPT_TEMPLATE.format(
                name=name,
                goal=goal,
                twist_phrase=twist
            ),
            "metadata": {
                "combination_index": idx,
                "protagonist_category": categorize_name(name),
                "goal_category": categorize_goal(goal),
                "twist_category": categorize_twist(twist)
            }
        }

        prompts.append(prompt_entry)

    # Create dataset structure
    dataset = {
        "system_prompt": "You are a creative fiction writer skilled at crafting short stories with engaging twists.",
        "prompts": prompts,
        "dataset_metadata": {
            "total_prompts": len(prompts),
            "total_names": len(names),
            "total_goals": len(goals),
            "total_twists": len(twist_phrases),
            "names": names,
            "goals": goals,
            "twist_phrases": twist_phrases
        }
    }

    # Save to YAML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✓ Generated {len(prompts)} prompt combinations")
    print(f"✓ Saved to: {output_file}")

    return dataset


def categorize_name(name: str) -> str:
    """Categorize name type (placeholder for future categorization)."""
    # TODO: Add logic to categorize names (e.g., common, uncommon, gender, cultural origin)
    return "uncategorized"


def categorize_goal(goal: str) -> str:
    """Categorize goal type (placeholder for future categorization)."""
    # TODO: Add logic to categorize goals (e.g., personal, professional, survival, relationship)
    return "uncategorized"


def categorize_twist(twist: str) -> str:
    """Categorize twist type (placeholder for future categorization)."""
    # TODO: Add logic to categorize twists (e.g., dream, reveal, reversal, supernatural)
    return "uncategorized"


def main():
    parser = argparse.ArgumentParser(
        description="Generate twist ablation dataset from NAME, GOAL, and TWIST_PHRASE combinations"
    )
    parser.add_argument(
        '--names',
        nargs='+',
        help='List of protagonist names (space-separated)',
        required=False
    )
    parser.add_argument(
        '--goals',
        nargs='+',
        help='List of protagonist goals (space-separated, use quotes for multi-word goals)',
        required=False
    )
    parser.add_argument(
        '--twists',
        nargs='+',
        help='List of twist phrases (space-separated, use quotes for multi-word twists)',
        required=False
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=TWIST_PROMPTS_YAML,
        help=f'Output YAML file (default: {TWIST_PROMPTS_YAML})'
    )

    args = parser.parse_args()

    # Default values if not provided (for testing)
    names = args.names if args.names else [
        "Alice", "Marcus", "Dr. Chen", "Luna", "The Traveler"
    ]

    goals = args.goals if args.goals else [
        "find the truth",
        "save their family",
        "complete the mission",
        "escape the maze",
        "discover their identity"
    ]

    twists = args.twists if args.twists else [
        "it was only a dream",
        "they were dead all along",
        "the villain was their future self",
        "the world is a simulation",
        "they are the last human alive"
    ]

    # Display what will be generated
    print(f"\n📊 Dataset Generation Configuration:")
    print(f"   Names ({len(names)}): {names}")
    print(f"   Goals ({len(goals)}): {goals}")
    print(f"   Twists ({len(twists)}): {twists}")
    print(f"   Total combinations: {len(names) * len(goals) * len(twists)}\n")

    # Generate dataset
    generate_dataset(names, goals, twists, args.output)


if __name__ == "__main__":
    main()
