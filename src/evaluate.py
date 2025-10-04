from config import BT_REWARD
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import json
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import hashlib

def evaluate(samples:List[dict], device: str = "cuda", batch_size: int = 8,
            tokenizer: AutoTokenizer = None,
            reward_model: AutoModelForSequenceClassification = None): 
    """
    Evaluate the samples based on the models and metrics.
    Uses batched inference with a DataLoader; control batch size via batch_size.

    Expects each sample to contain keys:
      - 'prompt_text': str
      - 'content': str
    """
    prompts = [s['prompt_text'] for s in samples]
    contents = [s['content'] for s in samples]
    scores = []

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BT_REWARD)

    if reward_model is None:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",   # can also use "fp4"
        bnb_4bit_compute_dtype="float16"
        )

        reward_model = AutoModelForSequenceClassification.from_pretrained(BT_REWARD, 
        quantization_config=bnb_config, device_map="auto")
        reward_model.eval()

    def collate(batch_items):
        texts = []
        for wp, gen in batch_items:
            texts.append(f"User:\n{wp}\n\nAssistant:\n{gen}")
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return encoded

    pairs = list(zip(prompts, contents))
    data_loader = DataLoader(pairs, batch_size=batch_size, shuffle=False, collate_fn=collate)

    with torch.no_grad():
        total_batches = (len(pairs) + batch_size - 1) // batch_size if batch_size > 0 else 0
        for batch in tqdm(data_loader, total=total_batches, desc="Scoring", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = reward_model(**batch).logits
            batch_scores = logits[:, 0].detach().cpu().tolist()
            scores.extend(batch_scores)
    
    return scores


def fictional_evaluation(WP: str, content: str,
                        reward_model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                        device: str = "cuda") -> float:
    """
    Score the content for a fictional story based on Bradley-Terry reward model.

    """
    prompt = f"User:\n{WP}\n\nAssistant:\n{content}"
    tokenized_text = tokenizer(prompt, return_tensors="pt").to(device)
    return reward_model(**tokenized_text).logits[0][0].item()


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        sha256.update(f.read())
    return sha256.hexdigest()


def load_existing_results(reward_file: str) -> Dict:
    """Load existing reward.json if it exists."""
    if os.path.exists(reward_file):
        try:
            with open(reward_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}



if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate samples from single JSON file or ablation_rollouts directory.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single JSON file or directory containing ablation rollouts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path (for single file mode only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=["cuda", "cpu"],
        help="Device to run models on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched scoring."
    )

    args = parser.parse_args()

    # Auto-detect device if not provided
    selected_device = args.device
    if selected_device is None:
        try:
            import torch  # type: ignore

            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            selected_device = "cpu"

    input_path = Path(args.input)

    # Initialize tokenizer and model once
    print("Loading tokenizer and reward model...")
    tokenizer = AutoTokenizer.from_pretrained(BT_REWARD)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(BT_REWARD,
        quantization_config=bnb_config, device_map="auto")
    reward_model.eval()
    print("Model loaded successfully.\n")

    # Case 1: Single JSON file
    if input_path.is_file() and input_path.suffix == '.json':
        print(f"Processing single file: {input_path}")

        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            prompt_text = data.get("prompt_text", "")
            # Handle both single response and rollout format
            if 'rollout_data' in data:
                content = data['rollout_data'].get('content', '')
            elif 'response' in data and 'responses' in data['response']:
                content = data['response']['responses'][0].get('content', '')
            else:
                print(f"Could not find content in JSON file", file=sys.stderr)
                sys.exit(2)

            samples = [{"prompt_text": prompt_text, "content": content}]
            scores = evaluate(samples, device=selected_device, batch_size=1,
                            tokenizer=tokenizer, reward_model=reward_model)

            result = {
                "file": str(input_path.name),
                "score": scores[0],
                "prompt_id": data.get("prompt_id", "unknown")
            }

            # Determine output path
            if args.output:
                output_file = Path(args.output)
            else:
                output_file = input_path.parent / f"{input_path.stem}_evaluated.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"✓ Evaluation saved to: {output_file}")
            print(f"  Score: {scores[0]:.4f}")

        except Exception as e:
            print(f"Failed to process file {input_path}: {e}", file=sys.stderr)
            sys.exit(2)

    # Case 2: Directory with ablation rollouts
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}\n")

        # Create output directory with _lmbench_evaluated suffix
        output_base = input_path.parent / f"{input_path.name}_lmbench_evaluated"
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_base}\n")

        # Find all directories containing rollout files
        rollout_dirs: List[Path] = []
        for root, dirs, files in os.walk(input_path):
            rollout_files = [f for f in files if f.startswith("rollout_") and f.endswith(".json")]
            if rollout_files:
                rollout_dirs.append(Path(root))

        if not rollout_dirs:
            print(f"No rollout files found under {input_path}", file=sys.stderr)
            sys.exit(2)

        print(f"Found {len(rollout_dirs)} directories with rollout files\n")

        created_outputs: List[str] = []
        total_evaluated = 0
        total_skipped = 0

        for rollout_dir in tqdm(sorted(rollout_dirs), desc="Processing directories"):
            try:
                filenames = [fn for fn in os.listdir(rollout_dir)
                           if fn.startswith("rollout_") and fn.endswith(".json")]
            except Exception as e:
                print(f"Failed to list directory {rollout_dir}: {e}", file=sys.stderr)
                continue

            if not filenames:
                continue

            # Create corresponding output directory structure
            rel_path = rollout_dir.relative_to(input_path)
            output_dir = output_base / rel_path
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load existing results
            output_path = output_dir / "reward.json"
            existing_results = load_existing_results(str(output_path))

            # Separate files into those needing evaluation and those already done
            samples_to_eval: List[dict] = []
            file_metadata_to_eval: List[Dict] = []
            new_results: Dict[str, Dict] = {}

            for fn in sorted(filenames):
                fpath = rollout_dir / fn
                rollout_key = os.path.splitext(fn)[0]

                try:
                    # Compute checksum
                    checksum = compute_file_checksum(str(fpath))

                    # Check if already evaluated with same checksum
                    if checksum in existing_results:
                        # Skip - already evaluated with same content
                        new_results[checksum] = existing_results[checksum]
                        total_skipped += 1
                        continue

                    # Need to evaluate this file
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    prompt_text = data.get("prompt_text", "")
                    content = data.get("rollout_data", {}).get("content", "")

                    # Prepare metadata
                    rel_file_path = fpath.relative_to(input_path)
                    metadata = {
                        "rollout_name": rollout_key,
                        "filename": fn,
                        "relative_path": str(rel_file_path),
                        "absolute_path": str(fpath.absolute()),
                        "checksum": checksum,
                        "prompt_id": data.get("prompt_id", "unknown"),
                        "strategy": data.get("strategy", "unknown"),
                        "think_mode": data.get("think_mode", "unknown"),
                        "para_index": data.get("para_index", -1),
                        "rollout_index": data.get("rollout_index", -1),
                    }

                    samples_to_eval.append({"prompt_text": prompt_text, "content": content})
                    file_metadata_to_eval.append(metadata)

                except Exception as e:
                    print(f"Failed to process {fpath}: {e}", file=sys.stderr)
                    continue

            # Evaluate only new/changed files
            if samples_to_eval:
                local_batch_size = min(args.batch_size, len(samples_to_eval))
                scores = evaluate(samples_to_eval, device=selected_device, batch_size=local_batch_size,
                                tokenizer=tokenizer, reward_model=reward_model)

                # Update results with new scores and metadata
                for metadata, score in zip(file_metadata_to_eval, scores):
                    checksum = metadata["checksum"]
                    new_results[checksum] = {
                        **metadata,
                        "score": score,
                    }
                    total_evaluated += 1

            # Write updated results
            try:
                with open(output_path, "w") as f:
                    json.dump(new_results, f, indent=2)
                created_outputs.append(str(output_path))
            except Exception as e:
                print(f"Failed to write output JSON to {output_path}: {e}", file=sys.stderr)
                continue

        print(f"\n{'='*80}")
        print(f"Evaluation complete!")
        print(f"Total evaluated: {total_evaluated}")
        print(f"Total skipped (unchanged): {total_skipped}")
        print(f"Processed {len(created_outputs)} directories")
        print(f"Output saved to: {output_base}")
        print(f"{'='*80}\n")

    else:
        print(f"Invalid input: {input_path} is neither a JSON file nor a directory", file=sys.stderr)
        sys.exit(1)