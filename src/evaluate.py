from config import MODEL_PATHS
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from tigerscore import TIGERScorer
import json
from typing import List

def evaluate(samples:dict, device: str = "cuda"): 
    """
    Evaluate the samples based on the models and metrics.
    
    """
    reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS.bt_reward).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS.bt_reward)
    scorer = TIGERScorer(model_name=MODEL_PATHS.tiger_score).to(device)
    
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