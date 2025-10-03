#!/usr/bin/env python3
"""
Count paragraphs and words in both reasoning and content fields from JSON response files.

Usage:
    python scripts/count_reasoning_paragraphs.py <directory_or_json_file>
    python scripts/count_reasoning_paragraphs.py data/fiction/descriptive_prompt/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import glob


def count_paragraphs(text: str) -> int:
    """
    Count paragraphs in text, treating double newlines as paragraph separators.
    """
    if not text:
        return 0

    # Split by double newlines (common paragraph separator)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)


def analyze_json_file(file_path: Path) -> Dict:
    """
    Analyze a JSON file and extract reasoning and content paragraph/word counts.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Navigate to responses[0]
    try:
        response = data['response']['responses'][0]
        reasoning = response.get('reasoning', '')
        content = response.get('content', '')

        return {
            'file': file_path.name,
            'prompt_id': data.get('prompt_id', 'N/A'),
            'reasoning_paragraphs': count_paragraphs(reasoning),
            'reasoning_words': len(reasoning.split()) if reasoning else 0,
            'content_paragraphs': count_paragraphs(content),
            'content_words': len(content.split()) if content else 0,
        }
    except (KeyError, IndexError, TypeError) as e:
        return {
            'file': file_path.name,
            'error': f"Could not extract data: {e}"
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/count_reasoning_paragraphs.py <directory_or_json_file> [...]")
        print("Examples:")
        print("  python scripts/count_reasoning_paragraphs.py data/fiction/descriptive_prompt/")
        print("  python scripts/count_reasoning_paragraphs.py data/fiction/descriptive_prompt/*.json")
        sys.exit(1)

    results = []
    json_files = []

    # Collect all JSON files from arguments (can be directories or files)
    for file_arg in sys.argv[1:]:
        path = Path(file_arg)

        if path.is_dir():
            # If it's a directory, find all JSON files in it
            json_files.extend(path.glob('*.json'))
        elif path.is_file() and path.suffix == '.json':
            # If it's a JSON file, add it
            json_files.append(path)
        elif '*' in str(file_arg):
            # If it's a glob pattern, expand it
            json_files.extend([Path(f) for f in glob.glob(file_arg)])
        else:
            print(f"Warning: Not a valid directory or JSON file: {path}")

    if not json_files:
        print("No JSON files found!")
        sys.exit(1)

    # Process all files
    for file_path in sorted(json_files):
        result = analyze_json_file(file_path)
        results.append(result)

    # Print results
    print(f"\n{'File':<35} {'ID':<8} {'Reasoning':<20} {'Content':<20}")
    print(f"{'':35} {'':8} {'Para':>6} {'Words':>12} {'Para':>6} {'Words':>12}")
    print("-" * 91)

    for result in results:
        if 'error' in result:
            print(f"{result['file']:<35} ERROR: {result['error']}")
        else:
            print(f"{result['file']:<35} {result['prompt_id']:<8} "
                  f"{result['reasoning_paragraphs']:>6} {result['reasoning_words']:>12} "
                  f"{result['content_paragraphs']:>6} {result['content_words']:>12}")

    # Print summary if multiple files
    if len(results) > 1 and not any('error' in r for r in results):
        valid_results = [r for r in results if 'reasoning_paragraphs' in r]
        avg_reasoning_para = sum(r['reasoning_paragraphs'] for r in valid_results) / len(valid_results)
        avg_reasoning_words = sum(r['reasoning_words'] for r in valid_results) / len(valid_results)
        avg_content_para = sum(r['content_paragraphs'] for r in valid_results) / len(valid_results)
        avg_content_words = sum(r['content_words'] for r in valid_results) / len(valid_results)

        print("-" * 91)
        print(f"Total files: {len(results)}")
        print(f"{'Averages:':<44} {avg_reasoning_para:>6.1f} {avg_reasoning_words:>12.1f} "
              f"{avg_content_para:>6.1f} {avg_content_words:>12.1f}")


if __name__ == "__main__":
    main()
