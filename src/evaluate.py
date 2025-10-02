from config import MODEL_PATHS
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from tigerscore import TIGERScorer
import json
from typing import List

def evaluate(samples:List[dict], device: str = "cuda"): 
    """
    Evaluate the samples based on the models and metrics.
    
    """
    # reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS.bt_reward).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS.bt_reward)
    print("Initialized scorer")
    scorer = TIGERScorer(model_name=MODEL_PATHS.tiger_score)
    
    for sample in samples:
        is_fictional = sample["is_fictional"]
        if is_fictional:
            score = fictional_evaluation(sample["WP"], sample["content"], reward_model, tokenizer, device)
        else:
            score = non_fictional_evaluation([sample["instruction"]], [sample["input_context"]], [sample["content"]], scorer, device)
        sample["score"] = score
    
    return samples


def fictional_evaluation(WP: str, content: str, 
                        reward_model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, 
                        device: str = "cuda") -> float:
    """
    Score the content for a fictional story based on Bradley-Terry reward model.
    
    """
    prompt = f"User:\n{WP}\n\nAssistant:\n{content}"
    tokenized_text = tokenizer(prompt, return_tensors="pt").to(device)
    return reward_model(**tokenized_text).logits[0][0].item()


def non_fictional_evaluation(instruction: List[str], input_context: List[str], content: List[str],
                             scorer: TIGERScorer, device: str = "cuda") -> float:
    """
    Score the content for a non-fictional story based on TIGERScore.
    
    """
    results = scorer.score(instruction, content, input_context)
    return results


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

    try:
        evaluated = evaluate(samples, device=selected_device)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(evaluated, f, indent=2)
        except Exception as e:
            print(f"Failed to write output JSON to {args.output}: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        print(json.dumps(evaluated, indent=2))