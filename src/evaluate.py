from config import MODEL_PATH, FICTIONAL_PROMPTS_YAML
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import json
from typing import List
import yaml
import torch
from torch.utils.data import DataLoader

def evaluate(samples:List[dict], device: str = "cuda", batch_size: int = 8): 
    """
    Evaluate the samples based on the models and metrics.
    Uses batched inference with a DataLoader; control batch size via batch_size.
    
    """
    with open(FICTIONAL_PROMPTS_YAML, "r") as f:
        data = yaml.safe_load(f)
        
    sample_ids = [s["sample_id"] - 1  for s in samples]

    prompts =  [data["prompts"][sid] for sid in sample_ids]

    contents = [s['response']['responses'][0]['content'] for s in samples]
    scores = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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
        for batch in data_loader:
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

    parser = argparse.ArgumentParser(description="Evaluate samples from a JSON file.")
    parser.add_argument(
        "--input",
        type=str,
        default="/pscratch/sd/r/ritesh11/temp/test.json",
        help="Path to input JSON file containing a list of samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write evaluated samples as JSON",
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

    try:
        with open(args.input, "r") as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to read input JSON from {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Monkeypatch to handle single sample at this moment
    if type(samples) == dict:
        samples = [samples]

    if args.batch_size > len(samples):
        args.batch_size = len(samples)
    
    evaluated = evaluate(samples, device=selected_device, batch_size=args.batch_size)


    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(evaluated, f, indent=2)
        except Exception as e:
            print(f"Failed to write output JSON to {args.output}: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        print(json.dumps(evaluated, indent=2))