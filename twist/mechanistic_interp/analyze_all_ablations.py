#!/usr/bin/env python3
"""
Batch Semantic Analysis for All Ablation Experiments

This script processes all ablation experiments and computes semantic similarities
for each baseline × think_mode combination.

Output: JSON files with 8 semantic similarity metrics per ablation pair.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re
from copy import deepcopy

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import OUTPUT_DIR_BASE_PATH

# Try to import embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
SEMANTIC_AVAILABLE = True

from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline
import torch
TRANSFORMERS_AVAILABLE = True


class QwenEmbeddings:
    """Wrapper to use Qwen model for embeddings via mean pooling."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], convert_to_numpy: bool = True):
        """Encode texts to embeddings using mean pooling."""
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()

            return embeddings


def load_semantic_model(model_type: str = "qwen"):
    """Load semantic similarity model."""
    if model_type.lower() in ['qwen', 'qwen3-0.6b', 'qwen/qwen3-0.6b']:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install: pip install transformers torch")
        print(f"Loading Qwen model for embeddings...")
        return QwenEmbeddings("Qwen/Qwen3-0.6B")
    else:
        if not SEMANTIC_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install: pip install sentence-transformers")
        print(f"Loading sentence-transformers model: {model_type}...")
        return SentenceTransformer(model_type)


def load_presence_model(model_name: str):
    """Load chat-style generator for twist presence detection."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not installed. Install: pip install transformers")

    print(f"Loading presence model: {model_name}...")
    generator = hf_pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=model_name
    )
    print("✓ Presence model loaded")
    return generator


def empty_presence() -> Dict[str, Dict[str, Optional[float]]]:
    """Return a fresh empty presence structure."""
    return {
        'clean': {'present': None, 'confidence': None},
        'corrupted': {'present': None, 'confidence': None}
    }


def _extract_presence_from_text(text: str) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Extract presence data from a JSON block in the model response."""

    if not text or "{" not in text:
        return None

    try:
        matches = re.findall(r"\{.*\}", text, re.DOTALL)
        if not matches:
            return None
        parsed = None
        for candidate in reversed(matches):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            return None

        clean = parsed.get('clean', {})
        corrupted = parsed.get('corrupted', {})

        def normalize(entry):
            present = entry.get('present')
            confidence = entry.get('confidence')
            if isinstance(present, str):
                present_lower = present.lower()
                if present_lower in {'true', 'yes', 'present'}:
                    present = True
                elif present_lower in {'false', 'no', 'absent'}:
                    present = False
                else:
                    present = None
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = None
            return {
                'present': present if isinstance(present, bool) else None,
                'confidence': confidence if isinstance(confidence, (int, float)) else None
            }

        return {
            'clean': normalize(clean),
            'corrupted': normalize(corrupted)
        }
    except Exception:
        return None


PRESENCE_PROMPT_TEMPLATE = (
    "You are checking whether a story contains specific narrative twists.\n"
    "Return a strict JSON object with this schema:\n"
    "{\n"
    "  \"clean\": {\"present\": true|false, \"confidence\": float between 0 and 1},\n"
    "  \"corrupted\": {\"present\": true|false, \"confidence\": float between 0 and 1}\n"
    "}.\n"
    "Confidence should reflect your certainty that the twist is explicitly or implicitly present.\n"
    "Clean twist: {clean_twist}\n"
    "Corrupted twist: {corrupted_twist}\n"
    "Story:\n"
    "{story}\n"
    "JSON:"
)


def build_presence_prompt(text: str, clean_twist: str, corrupted_twist: str) -> Optional[str]:
    """Construct the instruction prompt for twist presence checking."""
    text = (text or "").strip()
    if not text or not (clean_twist or corrupted_twist):
        return None

    return PRESENCE_PROMPT_TEMPLATE.format(
        clean_twist=clean_twist or 'N/A',
        corrupted_twist=corrupted_twist or 'N/A',
        story=text
    )


def parse_presence_result(raw_output: str, threshold: float = 0.5) -> Dict[str, Dict[str, Optional[float]]]:
    """Parse raw LLM output into presence scores."""

    default_presence = empty_presence()
    if not raw_output:
        return default_presence

    presence = _extract_presence_from_text(raw_output)
    if not presence:
        return default_presence

    for key in ['clean', 'corrupted']:
        entry = presence.get(key, {})
        present = entry.get('present')
        confidence = entry.get('confidence')
        if present is None and isinstance(confidence, (int, float)):
            entry['present'] = confidence >= threshold
        presence[key] = entry

    # Ensure both keys exist
    for key in ['clean', 'corrupted']:
        presence.setdefault(key, {'present': None, 'confidence': None})
        presence[key].setdefault('present', None)
        presence[key].setdefault('confidence', None)

    return presence


def run_presence_batch(generator, prompts: List[str], threshold: float = 0.5) -> List[Dict[str, Dict[str, Optional[float]]]]:
    """Run presence model on a batch of prompts."""

    if generator is None or not prompts:
        return [empty_presence() for _ in prompts]

    try:
        outputs = generator(
            prompts,
            max_new_tokens=2000,
            do_sample=False,
            return_full_text=False
        )
    except Exception as exc:
        print(f"  ⚠ Presence generation failed: {exc}")
        return [empty_presence() for _ in prompts]

    results = []
    for output in outputs:
        raw_output = output
        if isinstance(output, list):
            raw_output = output[0] if output else {}
        if isinstance(raw_output, dict):
            raw_output = raw_output.get('generated_text') or raw_output.get('text') or ''
        if isinstance(raw_output, list):
            raw_output = raw_output[0] if raw_output else ''
        results.append(parse_presence_result(raw_output, threshold))

    # In case pipeline returns fewer outputs than prompts, pad with defaults
    while len(results) < len(prompts):
        results.append(empty_presence())

    return results


def compute_semantic_similarity(text1, text2, model) -> float:
    """Compute cosine similarity between two texts."""
    if model is None:
        return 0.0

    # Ensure both inputs are strings
    if not isinstance(text1, str):
        text1 = str(text1) if text1 is not None else ""
    if not isinstance(text2, str):
        text2 = str(text2) if text2 is not None else ""

    # Handle empty strings
    if not text1 or not text2:
        return 0.0

    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    emb1, emb2 = embeddings[0], embeddings[1]

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return float(similarity)


def load_baseline_story(baseline_id: str, stories_dir: Path) -> Dict:
    """Load baseline story."""
    baseline_file = stories_dir / f"{baseline_id}.json"
    with open(baseline_file, 'r') as f:
        return json.load(f)


def load_ablation_rollout(
    baseline_id: str,
    target_id: str,
    think_mode: str,
    ablation_dir: Path
) -> Dict:
    """Load ablation rollout."""
    rollout_file = (
        ablation_dir / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )

    if rollout_file.exists():
        with open(rollout_file, 'r') as f:
            return json.load(f)
    return None


def analyze_single_ablation(
    baseline_data: Dict,
    ablation_data: Dict,
    semantic_model,
    presence_classifier=None,
    presence_threshold: float = 0.5
) -> Dict:
    """Compute 8 semantic similarities for one ablation pair."""

    # Extract data
    baseline_story = baseline_data['response']['responses'][0]['content']
    ablation_story = ablation_data['rollout_data']['content']

    baseline_reasoning = baseline_data['response']['responses'][0]['reasoning']

    # For ablation reasoning:
    # - no_more_thinking: skip computing ablation reasoning metrics
    # - allow_more_thinking: reasoning is new (model generated)
    ablation_reasoning = ablation_data['rollout_data'].get('reasoning')
    think_mode = ablation_data['ablation_metadata']['think_mode']
    analyze_reasoning = (think_mode == 'allow_more_thinking')

    clean_twist = baseline_data['twist_phrase']
    corrupted_twist = ablation_data['new_twist']

    # Compute semantic similarities
    baseline_content_vs_clean = compute_semantic_similarity(
        baseline_story, clean_twist, semantic_model
    )
    baseline_content_vs_corrupted = compute_semantic_similarity(
        baseline_story, corrupted_twist, semantic_model
    )
    ablation_content_vs_clean = compute_semantic_similarity(
        ablation_story, clean_twist, semantic_model
    )
    ablation_content_vs_corrupted = compute_semantic_similarity(
        ablation_story, corrupted_twist, semantic_model
    )
    baseline_reasoning_vs_clean = compute_semantic_similarity(
        baseline_reasoning, clean_twist, semantic_model
    )
    baseline_reasoning_vs_corrupted = compute_semantic_similarity(
        baseline_reasoning, corrupted_twist, semantic_model
    )

    result = {
        'baseline_id': baseline_data['prompt_id'],
        'target_id': ablation_data['twist_prompt_id'],
        'think_mode': think_mode,
        'clean_twist': clean_twist,
        'corrupted_twist': corrupted_twist,
        'baseline_content_vs_clean': baseline_content_vs_clean,
        'baseline_content_vs_corrupted': baseline_content_vs_corrupted,
        'ablation_content_vs_clean': ablation_content_vs_clean,
        'ablation_content_vs_corrupted': ablation_content_vs_corrupted,
        'baseline_reasoning_vs_clean': baseline_reasoning_vs_clean,
        'baseline_reasoning_vs_corrupted': baseline_reasoning_vs_corrupted,
    }

    # Determine content preference
    result['content_prefers'] = 'clean' if ablation_content_vs_clean > ablation_content_vs_corrupted else 'corrupted'

    # Reasoning metrics only for allow_more_thinking
    if analyze_reasoning and ablation_reasoning is not None:
        ablation_reasoning_vs_clean = compute_semantic_similarity(
            ablation_reasoning, clean_twist, semantic_model
        )
        ablation_reasoning_vs_corrupted = compute_semantic_similarity(
            ablation_reasoning, corrupted_twist, semantic_model
        )
        result['ablation_reasoning_vs_clean'] = ablation_reasoning_vs_clean
        result['ablation_reasoning_vs_corrupted'] = ablation_reasoning_vs_corrupted
        result['reasoning_prefers'] = 'clean' if ablation_reasoning_vs_clean > ablation_reasoning_vs_corrupted else 'corrupted'
    else:
        result['ablation_reasoning_vs_clean'] = None
        result['ablation_reasoning_vs_corrupted'] = None
        result['reasoning_prefers'] = None

    default_presence = empty_presence()
    result['baseline_presence'] = deepcopy(default_presence)
    result['ablation_presence'] = deepcopy(default_presence)
    result['baseline_reasoning_presence'] = deepcopy(default_presence)
    result['ablation_reasoning_presence'] = deepcopy(default_presence)

    if presence_classifier is not None:
        presence_prompts = {
            'baseline_presence': build_presence_prompt(baseline_story, clean_twist, corrupted_twist),
            'ablation_presence': build_presence_prompt(ablation_story, clean_twist, corrupted_twist),
            'baseline_reasoning_presence': build_presence_prompt(baseline_reasoning, clean_twist, corrupted_twist),
            'ablation_reasoning_presence': build_presence_prompt(ablation_reasoning if analyze_reasoning else None, clean_twist, corrupted_twist),
        }
        result['_presence_prompts'] = presence_prompts

    return result


def analyze_baseline_think_mode(
    baseline_id: str,
    think_mode: str,
    semantic_model,
    stories_dir: Path,
    ablation_dir: Path,
    presence_classifier=None,
    presence_threshold: float = 0.5
) -> Dict:
    """Analyze all ablations for one baseline × think_mode combination."""

    print(f"\nAnalyzing {baseline_id} × {think_mode}")
    print("=" * 80)

    # Load baseline
    baseline_data = load_baseline_story(baseline_id, stories_dir)

    # Find all target ablations
    baseline_ablation_dir = ablation_dir / f"baseline_{baseline_id}"

    if not baseline_ablation_dir.exists():
        print(f"  ⚠ No ablations found for {baseline_id}")
        return None

    # Process each target
    ablations = []
    for target_dir in sorted(baseline_ablation_dir.glob("target_*")):
        target_id = target_dir.name.replace("target_", "")

        # Load ablation rollout
        ablation_data = load_ablation_rollout(
            baseline_id, target_id, think_mode, ablation_dir
        )

        if ablation_data is None:
            print(f"  ⚠ Skipping {target_id} - no rollout found")
            continue

        # Analyze
        result = analyze_single_ablation(
            baseline_data,
            ablation_data,
            semantic_model,
            presence_classifier=presence_classifier,
            presence_threshold=presence_threshold
        )
        ablations.append(result)

        reasoning_str = result['reasoning_prefers'] if result['reasoning_prefers'] is not None else 'skipped'
        print(f"  ✓ {target_id}: content={result['content_prefers']}, reasoning={reasoning_str}")

    # Run presence model in batch (if available)
    if presence_classifier is not None and ablations:
        prompts = []
        index_map = []

        for result in ablations:
            prompts_map = result.pop('_presence_prompts', {}) if '_presence_prompts' in result else {}
            for field, prompt in prompts_map.items():
                if prompt:
                    prompts.append(prompt)
                    index_map.append((result, field))

        if prompts:
            presence_results = run_presence_batch(presence_classifier, prompts, threshold=presence_threshold)
            for (result, field), presence in zip(index_map, presence_results):
                result[field] = presence

    # Compute summary statistics
    if ablations:
        content_prefers_clean = sum(1 for a in ablations if a['content_prefers'] == 'clean')

        # Common aggregates
        summary = {
            'total_ablations': len(ablations),
            'content_prefers_clean': content_prefers_clean,
            'content_prefers_corrupted': len(ablations) - content_prefers_clean,
            'avg_baseline_content_vs_clean': sum(a['baseline_content_vs_clean'] for a in ablations) / len(ablations),
            'avg_baseline_content_vs_corrupted': sum(a['baseline_content_vs_corrupted'] for a in ablations) / len(ablations),
            'avg_ablation_content_vs_clean': sum(a['ablation_content_vs_clean'] for a in ablations) / len(ablations),
            'avg_ablation_content_vs_corrupted': sum(a['ablation_content_vs_corrupted'] for a in ablations) / len(ablations),
            'avg_baseline_reasoning_vs_clean': sum(a['baseline_reasoning_vs_clean'] for a in ablations) / len(ablations),
            'avg_baseline_reasoning_vs_corrupted': sum(a['baseline_reasoning_vs_corrupted'] for a in ablations) / len(ablations),
        }

        if think_mode == 'allow_more_thinking':
            reasoning_prefers_clean = sum(1 for a in ablations if a['reasoning_prefers'] == 'clean')
            # Averages for ablation reasoning (valid only in allow_more_thinking)
            valid_clean = [a['ablation_reasoning_vs_clean'] for a in ablations if a['ablation_reasoning_vs_clean'] is not None]
            valid_corrupt = [a['ablation_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            avg_ablation_reasoning_vs_clean = sum(valid_clean) / len(valid_clean) if valid_clean else None
            avg_ablation_reasoning_vs_corrupted = sum(valid_corrupt) / len(valid_corrupt) if valid_corrupt else None

            # Correlation: increase in similarity (new twist vs reasoning) vs old reasoning similarity
            base_corr_list = [a['baseline_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            delta_corr_list = [a['ablation_reasoning_vs_corrupted'] - a['baseline_reasoning_vs_corrupted'] for a in ablations if a['ablation_reasoning_vs_corrupted'] is not None]
            corr_delta_vs_old = None
            try:
                if len(base_corr_list) > 1:
                    import numpy as _np
                    if _np.std(base_corr_list) > 0 and _np.std(delta_corr_list) > 0:
                        corr_delta_vs_old = float(_np.corrcoef(base_corr_list, delta_corr_list)[0, 1])
            except Exception:
                corr_delta_vs_old = None

            summary.update({
                'reasoning_prefers_clean': reasoning_prefers_clean,
                'reasoning_prefers_corrupted': len(ablations) - reasoning_prefers_clean,
                'avg_ablation_reasoning_vs_clean': avg_ablation_reasoning_vs_clean,
                'avg_ablation_reasoning_vs_corrupted': avg_ablation_reasoning_vs_corrupted,
                'corr_delta_reasoning_vs_old_reasoning': corr_delta_vs_old,
            })
        else:
            # no_more_thinking: reasoning metrics are skipped
            summary.update({
                'reasoning_prefers_clean': None,
                'reasoning_prefers_corrupted': None,
                'avg_ablation_reasoning_vs_clean': None,
                'avg_ablation_reasoning_vs_corrupted': None,
                'corr_delta_reasoning_vs_old_reasoning': None,
            })

        if presence_classifier is not None:
            def summarize_presence_field(field: str) -> Dict[str, Dict[str, Optional[float]]]:
                summary_data = {
                    'clean': {'present_count': 0, 'total': 0, 'confidence_avg': None},
                    'corrupted': {'present_count': 0, 'total': 0, 'confidence_avg': None}
                }

                clean_confidences = []
                corrupt_confidences = []

                for ablation in ablations:
                    presence = ablation.get(field) or empty_presence()

                    clean_present = presence['clean'].get('present')
                    if clean_present is not None:
                        summary_data['clean']['total'] += 1
                        if clean_present:
                            summary_data['clean']['present_count'] += 1
                    clean_conf = presence['clean'].get('confidence')
                    if isinstance(clean_conf, (int, float)):
                        clean_confidences.append(clean_conf)

                    corrupt_present = presence['corrupted'].get('present')
                    if corrupt_present is not None:
                        summary_data['corrupted']['total'] += 1
                        if corrupt_present:
                            summary_data['corrupted']['present_count'] += 1
                    corrupt_conf = presence['corrupted'].get('confidence')
                    if isinstance(corrupt_conf, (int, float)):
                        corrupt_confidences.append(corrupt_conf)

                if clean_confidences:
                    summary_data['clean']['confidence_avg'] = sum(clean_confidences) / len(clean_confidences)
                if corrupt_confidences:
                    summary_data['corrupted']['confidence_avg'] = sum(corrupt_confidences) / len(corrupt_confidences)

                return summary_data

            presence_summary = {
                'baseline_story': summarize_presence_field('baseline_presence'),
                'ablation_story': summarize_presence_field('ablation_presence'),
                'baseline_reasoning': summarize_presence_field('baseline_reasoning_presence'),
                'ablation_reasoning': summarize_presence_field('ablation_reasoning_presence'),
            }

            summary['presence'] = presence_summary

        print(f"\n  Summary:")
        print(f"    Content prefers clean: {content_prefers_clean}/{len(ablations)} ({100*content_prefers_clean/len(ablations):.1f}%)")
        if think_mode == 'allow_more_thinking':
            print(f"    Reasoning prefers clean: {summary['reasoning_prefers_clean']}/{len(ablations)} ({100*summary['reasoning_prefers_clean']/len(ablations):.1f}%)")
        if presence_classifier is not None and summary.get('presence'):
            presence_labels = {
                'baseline_story': 'Baseline story',
                'ablation_story': 'Ablation story',
                'baseline_reasoning': 'Baseline reasoning',
                'ablation_reasoning': 'Ablation reasoning',
            }
            print("    Twist presence (LLM check):")
            for key, label in presence_labels.items():
                data = summary['presence'].get(key)
                if not data:
                    print(f"      {label}: N/A")
                    continue

                clean = data['clean']
                corrupt = data['corrupted']

                if clean['total']:
                    clean_pct = 100 * clean['present_count'] / clean['total']
                    clean_conf = clean['confidence_avg']
                    clean_conf_str = f", avg conf {clean_conf:.2f}" if clean_conf is not None else ""
                    print(
                        f"      {label} – clean: {clean['present_count']}/{clean['total']}"
                        f" ({clean_pct:.1f}%{clean_conf_str})"
                    )
                else:
                    print(f"      {label} – clean: N/A")

                if corrupt['total']:
                    corrupt_pct = 100 * corrupt['present_count'] / corrupt['total']
                    corrupt_conf = corrupt['confidence_avg']
                    corrupt_conf_str = f", avg conf {corrupt_conf:.2f}" if corrupt_conf is not None else ""
                    print(
                        f"      {label} – corrupted: {corrupt['present_count']}/{corrupt['total']}"
                        f" ({corrupt_pct:.1f}%{corrupt_conf_str})"
                    )
                else:
                    print(f"      {label} – corrupted: N/A")
    else:
        summary = {}

    return {
        'baseline_id': baseline_id,
        'think_mode': think_mode,
        'ablations': ablations,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch semantic analysis for all ablation experiments'
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=['twist_001', 'twist_002', 'twist_003', 'twist_004', 'twist_005'],
        help='Baseline IDs to analyze (default: twist_001-005)'
    )
    parser.add_argument(
        '--think-modes',
        nargs='+',
        choices=['no_more_thinking', 'allow_more_thinking'],
        default=['no_more_thinking', 'allow_more_thinking'],
        help='Think modes to analyze'
    )
    parser.add_argument(
        '--semantic-model',
        type=str,
        default='qwen',
        help='Embedding model (default: qwen)'
    )
    parser.add_argument(
        '--presence-model',
        type=str,
        default=None,
        help='Zero-shot HF model for twist presence detection (e.g., facebook/bart-large-mnli)'
    )
    parser.add_argument(
        '--presence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for marking twist presence'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('twist/mechanistic_interp/outputs/batch_analysis'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Semantic Analysis - All Ablation Experiments")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Baselines: {args.baselines}")
    print(f"  Think modes: {args.think_modes}")
    print(f"  Semantic model: {args.semantic_model}")
    print(f"  Output dir: {args.output_dir}")
    if args.presence_model:
        print(f"  Presence model: {args.presence_model}")
        print(f"  Presence threshold: {args.presence_threshold}")

    # Load semantic model
    print(f"\nLoading semantic model...")
    semantic_model = load_semantic_model(args.semantic_model)
    print(f"✓ Model loaded")

    presence_classifier = None
    if args.presence_model:
        presence_classifier = load_presence_model(args.presence_model)

    # Paths
    stories_dir = OUTPUT_DIR_BASE_PATH / "stories"
    ablation_dir = OUTPUT_DIR_BASE_PATH / "ablation_outputs"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each baseline × think_mode
    all_results = []

    for baseline_id in args.baselines:
        for think_mode in args.think_modes:
            result = analyze_baseline_think_mode(
                baseline_id,
                think_mode,
                semantic_model,
                stories_dir,
                ablation_dir,
                presence_classifier=presence_classifier,
                presence_threshold=args.presence_threshold
            )

            if result:
                all_results.append(result)

                # Save individual result
                output_file = args.output_dir / f"{baseline_id}_{think_mode}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n  ✓ Saved: {output_file}")

    # Save combined results
    combined_file = args.output_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Batch analysis complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Individual files: {len(all_results)}")
    print(f"  Combined file: {combined_file}")
    print(f"\nNext step:")
    print(f"  python twist/mechanistic_interp/cross_twist_analysis.py")


if __name__ == "__main__":
    main()
