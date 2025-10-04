from config import MODEL_PATH
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import json
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if reward_model is None:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",   # can also use "fp4"
        bnb_4bit_compute_dtype="float16"
        )

        reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, 
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



if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate samples from rollouts directory.")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to ablation_rollouts directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write a single JSON if only one target dir is found; otherwise results are written as reward.json in each allow_more_thinking directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=["cuda", "cpu"],
        help="Device to run models on. Defaults to auto-detect.",
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

    # Find all target directories named 'allow_more_thinking' under the input
    target_dirs: List[str] = []
    for root, dirs, files in os.walk(args.input):
        if os.path.basename(root) == "allow_more_thinking":
            target_dirs.append(root)

    if not target_dirs:
        print(f"No 'allow_more_thinking' directories found under {args.input}", file=sys.stderr)
        sys.exit(2)

    # Initialize tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, 
    quantization_config=bnb_config, device_map="auto")
    reward_model.eval()

    created_outputs: List[str] = []
    for tdir in tqdm(sorted(target_dirs), desc="Directories"):
        try:
            filenames = [fn for fn in os.listdir(tdir) if fn.startswith("rollout_") and fn.endswith(".json")]
        except Exception as e:
            print(f"Failed to list directory {tdir}: {e}", file=sys.stderr)
            sys.exit(2)

        if not filenames:
            # Skip directories without rollout files
            continue

        samples: List[dict] = []
        rollout_keys: List[str] = []
        for fn in tqdm(sorted(filenames), desc="Rollouts", leave=False):
            fpath = os.path.join(tdir, fn)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                prompt_text = data.get("prompt_text", "")
                content = data.get("rollout_data", {}).get("content", "")
                samples.append({"prompt_text": prompt_text, "content": content})
                rollout_keys.append(os.path.splitext(fn)[0])
            except Exception as e:
                print(f"Failed to read {fpath}: {e}", file=sys.stderr)
                sys.exit(2)

        local_batch_size = min(args.batch_size, len(samples))
        scores = evaluate(samples, device=selected_device, batch_size=local_batch_size,
                          tokenizer=tokenizer, reward_model=reward_model)
        rewards = {key: score for key, score in zip(rollout_keys, scores)}

        output_path = os.path.join(tdir, "reward.json")
        try:
            with open(output_path, "w") as f:
                json.dump(rewards, f, indent=2)
            created_outputs.append(output_path)
        except Exception as e:
            print(f"Failed to write output JSON to {output_path}: {e}", file=sys.stderr)
            sys.exit(3)

    # If a custom output path is provided and only one target dir processed, also write there
    if args.output and len(created_outputs) == 1:
        try:
            with open(created_outputs[0], "r") as f:
                data = json.load(f)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to write output JSON to {args.output}: {e}", file=sys.stderr)
            sys.exit(3)