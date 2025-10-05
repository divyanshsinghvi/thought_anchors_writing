#!/usr/bin/env python3
"""
Experiment #10 (nnterp-style): Low-memory residual stream patching via PyTorch hooks

This script mirrors the NNsight experiment but uses direct forward hooks to:
1) Capture residual at a chosen layer/location from a source (no_more_thinking)
2) Patch that residual into the target (allow_more_thinking) during a single forward
3) Measure clean vs corrupted keyword probabilities at the next token

Locations:
 - pre_attn:  residual entering attention sublayer
 - post_attn: residual entering MLP sublayer (after attn + residual)
 - post_mlp:  residual exiting the block (layer output)

Usage examples:
  python twist/mechanistic_interp/experiment_10_nnterp.py --pair 1 --model Qwen/Qwen3-0.6b --location post_mlp --max-input-tokens 512
  python twist/mechanistic_interp/experiment_10_nnterp.py --pair 1 --model Qwen/Qwen3-0.6b --location post_attn --layer 12
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twist.config import OUTPUT_DIR_BASE_PATH


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ABLATION_DIR = OUTPUT_DIR_BASE_PATH / "ablation_outputs"
OUTPUT_DIR = Path("twist/mechanistic_interp/outputs/experiment_10")

# Matched pairs (same as NNsight version)
MATCHED_PAIRS = [
    {
        "pair_id": 1,
        "baseline_id": "twist_001",
        "target_id": "twist_004",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
        "corrupted_keywords": [
            "simulation", "simulated", "simulator", "virtual",
            "matrix", "glitch", "code", "program"
        ],
    },
    {
        "pair_id": 2,
        "baseline_id": "twist_001",
        "target_id": "twist_005",
        "clean_twist": "it was only a dream",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["dream", "woke", "awoke", "asleep"],
        "corrupted_keywords": [
            "last human", "last person", "only human", "only survivor",
            "sole human", "alone", "extinct"
        ],
    },
    {
        "pair_id": 3,
        "baseline_id": "twist_004",
        "target_id": "twist_005",
        "clean_twist": "the world is a simulation",
        "corrupted_twist": "they are the last human alive",
        "clean_keywords": ["simulation", "glitch", "code", "fractal", "static"],
        "corrupted_keywords": [
            "last human", "last person", "only human", "only survivor",
            "sole human", "alone", "extinct"
        ],
    },
    {
        "pair_id": 4,
        "baseline_id": "twist_005",
        "target_id": "twist_004",
        "clean_twist": "they are the last human alive",
        "corrupted_twist": "the world is a simulation",
        "clean_keywords": ["last human", "last person", "alone", "extinct"],
        "corrupted_keywords": [
            "simulation", "simulated", "simulator", "virtual",
            "matrix", "glitch", "code", "program"
        ],
    }
]


def load_prompts(baseline_id: str, target_id: str, think_mode: str) -> str:
    rollout_file = (
        ABLATION_DIR / f"baseline_{baseline_id}" / f"target_{target_id}" /
        think_mode / "rollout_000.json"
    )
    with open(rollout_file, 'r') as f:
        data = json.load(f)
    return data['ablated_prompt']


def encode(tokenizer: AutoTokenizer, prompt: str, device: str, max_tokens: Optional[int]) -> Dict[str, torch.Tensor]:
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens
    ).to(device)


def avg_first_token_prob(probs: torch.Tensor, tokenizer: AutoTokenizer, words: List[str]) -> float:
    ids = []
    for w in words:
        tok = tokenizer.encode(w, add_special_tokens=False)
        if tok:
            ids.append(tok[0])
    if not ids:
        return 0.0
    idx = torch.tensor(ids, dtype=torch.long, device=probs.device)
    vals = probs.index_select(0, idx)
    return float(vals.mean().item())


def run_probs(model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits[0, -1, :], dim=0)
    return probs


def ensure_special_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    if tokenizer.eos_token_id is None and hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
        tokenizer.eos_token_id = model.config.eos_token_id
    if tokenizer.pad_token_id is None:
        # fall back to eos as pad
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            model.resize_token_embeddings(len(tokenizer))


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0
) -> str:
    ensure_special_tokens(tokenizer, model)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-6) if temperature > 0.0 else 1.0,
        top_p=top_p if temperature > 0.0 else 1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            **gen_kwargs
        )
    in_len = inputs['input_ids'].shape[1]
    new_tokens = out_ids[0, in_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_keyword_patterns(keywords: List[str]) -> List[Tuple[str, str]]:
    """Return list of (keyword, regex_pattern) pairs for counting.
    Uses simple word-boundary match on lowercase; also tries space-prefixed variants.
    """
    import re
    patterns = []
    for kw in keywords:
        kw_l = kw.strip().lower()
        if not kw_l:
            continue
        # word boundary pattern
        pat = rf"\b{re.escape(kw_l)}\b"
        patterns.append((kw_l, pat))
        # also add plural-free/space version not strictly needed when using \b
    return patterns


def count_keywords(text: str, keywords: List[str]) -> int:
    import re
    txt = text.lower()
    total = 0
    for kw, pat in build_keyword_patterns(keywords):
        total += len(re.findall(pat, txt))
    return total


def capture_residual_post_mlp(block: torch.nn.Module, storage: Dict):
    def hook(module, inp, out):
        storage['tensor'] = out.detach()
        return out
    return hook


def patch_residual_post_mlp(storage: Dict):
    def hook(module, inp, out):
        src = storage.get('tensor')
        if src is None:
            return out
        min_len = min(out.shape[1], src.shape[1])
        out = out.clone()
        out[:, :min_len, :] = src[:, :min_len, :]
        return out
    return hook


def capture_residual_pre(module_name: str, storage: Dict):
    def pre_hook(module, inputs):
        hidden = inputs[0]
        storage['tensor'] = hidden.detach()
        return inputs
    return pre_hook


def patch_residual_pre(storage: Dict):
    def pre_hook(module, inputs):
        hidden = inputs[0]
        src = storage.get('tensor')
        if src is None:
            return inputs
        min_len = min(hidden.shape[1], src.shape[1])
        hidden = hidden.clone()
        hidden[:, :min_len, :] = src[:, :min_len, :]
        new_inputs = (hidden,) + tuple(inputs[1:])
        return new_inputs
    return pre_hook


def run_layer_patch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    source_inputs: Dict[str, torch.Tensor],
    target_inputs: Dict[str, torch.Tensor],
    layer: int,
    location: str,
    clean_keywords: List[str],
    corrupted_keywords: List[str],
    metric: str = 'counts',
    gen_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
    samples: int = 1,
    dump_texts_dir: Optional[Path] = None
) -> Dict:
    # Baseline on target (support multi-sample averaging for counts)
    baseline_clean = 0.0
    baseline_corr = 0.0
    base_presence_clean = 0
    base_presence_corr = 0
    baseline_texts: List[str] = []
    if metric == 'counts':
        reps = max(1, samples)
        for i in range(reps):
            txt = generate_text(
                model, tokenizer, target_inputs,
                max_new_tokens=gen_tokens, temperature=temperature, top_p=top_p
            )
            c_clean = count_keywords(txt, clean_keywords)
            c_corr = count_keywords(txt, corrupted_keywords)
            baseline_clean += c_clean
            baseline_corr += c_corr
            base_presence_clean += int(c_clean > 0)
            base_presence_corr += int(c_corr > 0)
            if dump_texts_dir is not None:
                baseline_texts.append(txt)
        baseline_clean /= reps
        baseline_corr /= reps
    else:
        base_probs = run_probs(model, target_inputs).to('cpu')
        baseline_clean = avg_first_token_prob(base_probs, tokenizer, clean_keywords)
        baseline_corr  = avg_first_token_prob(base_probs, tokenizer, corrupted_keywords)

    storage: Dict[str, torch.Tensor] = {'tensor': None}

    # Capture from source
    if location == 'post_mlp':
        h_cap = model.model.layers[layer].register_forward_hook(capture_residual_post_mlp(model.model.layers[layer], storage))
        _ = run_probs(model, source_inputs)
        h_cap.remove()
        # Patch into target
        h_patch = model.model.layers[layer].register_forward_hook(patch_residual_post_mlp(storage))
        if metric == 'counts':
            # average multiple patched generations
            reps = max(1, samples)
            patched_clean_acc = 0.0
            patched_corr_acc = 0.0
            pat_presence_clean = 0
            pat_presence_corr = 0
            patched_texts: List[str] = []
            for i in range(reps):
                txt = generate_text(
                    model, tokenizer, target_inputs,
                    max_new_tokens=gen_tokens, temperature=temperature, top_p=top_p
                )
                c_clean = count_keywords(txt, clean_keywords)
                c_corr = count_keywords(txt, corrupted_keywords)
                patched_clean_acc += c_clean
                patched_corr_acc += c_corr
                pat_presence_clean += int(c_clean > 0)
                pat_presence_corr += int(c_corr > 0)
                if dump_texts_dir is not None:
                    patched_texts.append(txt)
            patched_clean = patched_clean_acc / reps
            patched_corr  = patched_corr_acc / reps
        else:
            patched_probs = run_probs(model, target_inputs).to('cpu')
        h_patch.remove()
    elif location == 'pre_attn':
        h_cap = model.model.layers[layer].self_attn.register_forward_pre_hook(capture_residual_pre('pre_attn', storage))
        _ = run_probs(model, source_inputs)
        h_cap.remove()
        h_patch = model.model.layers[layer].self_attn.register_forward_pre_hook(patch_residual_pre(storage))
        if metric == 'counts':
            reps = max(1, samples)
            patched_clean_acc = 0.0
            patched_corr_acc = 0.0
            pat_presence_clean = 0
            pat_presence_corr = 0
            patched_texts = []
            for i in range(reps):
                txt = generate_text(
                    model, tokenizer, target_inputs,
                    max_new_tokens=gen_tokens, temperature=temperature, top_p=top_p
                )
                c_clean = count_keywords(txt, clean_keywords)
                c_corr = count_keywords(txt, corrupted_keywords)
                patched_clean_acc += c_clean
                patched_corr_acc += c_corr
                pat_presence_clean += int(c_clean > 0)
                pat_presence_corr += int(c_corr > 0)
                if dump_texts_dir is not None:
                    patched_texts.append(txt)
            patched_clean = patched_clean_acc / reps
            patched_corr  = patched_corr_acc / reps
        else:
            patched_probs = run_probs(model, target_inputs).to('cpu')
        h_patch.remove()
    elif location == 'post_attn':
        h_cap = model.model.layers[layer].mlp.register_forward_pre_hook(capture_residual_pre('post_attn', storage))
        _ = run_probs(model, source_inputs)
        h_cap.remove()
        h_patch = model.model.layers[layer].mlp.register_forward_pre_hook(patch_residual_pre(storage))
        patched_probs = run_probs(model, target_inputs).to('cpu')
        h_patch.remove()
    else:
        raise ValueError("location must be one of: pre_attn, post_attn, post_mlp")

    if metric != 'counts':
        patched_clean = avg_first_token_prob(patched_probs, tokenizer, clean_keywords)
        patched_corr  = avg_first_token_prob(patched_probs, tokenizer, corrupted_keywords)

    result = {
        'layer': layer,
        'location': location,
        'baseline_clean_prob': float(baseline_clean),
        'baseline_corrupted_prob': float(baseline_corr),
        'patched_clean_prob': float(patched_clean),
        'patched_corrupted_prob': float(patched_corr),
        'delta_clean': float(patched_clean - baseline_clean),
        'delta_corrupted': float(patched_corr - baseline_corr),
        'baseline_corr_minus_clean': float(baseline_corr - baseline_clean),
        'patched_corr_minus_clean': float(patched_corr - patched_clean),
        'delta_corr_minus_clean': float((patched_corr - patched_clean) - (baseline_corr - baseline_clean)),
    }

    if metric == 'counts' and samples > 1:
        result.update({
            'baseline_presence_clean_rate': (base_presence_clean / samples),
            'baseline_presence_corrupted_rate': (base_presence_corr / samples),
            'patched_presence_clean_rate': (pat_presence_clean / samples),
            'patched_presence_corrupted_rate': (pat_presence_corr / samples),
            'delta_presence_score': ((pat_presence_corrupted_rate := (pat_presence_corr / samples)) - (pat_presence_clean_rate := (pat_presence_clean / samples))) - ((base_presence_corr / samples) - (base_presence_clean / samples))
        })

    # Optionally dump texts
    if dump_texts_dir is not None and metric == 'counts':
        dump_texts_dir.mkdir(parents=True, exist_ok=True)
        # dump only first sample when samples>1 to save disk
        try:
            if baseline_texts:
                (dump_texts_dir / f"layer{layer:02d}_baseline.txt").write_text(baseline_texts[0])
            if 'patched_texts' in locals() and patched_texts:
                (dump_texts_dir / f"layer{layer:02d}_patched.txt").write_text(patched_texts[0])
        except Exception:
            pass

    return result

    return {
        'layer': layer,
        'location': location,
        'baseline_clean_prob': baseline_clean,
        'baseline_corrupted_prob': baseline_corr,
        'patched_clean_prob': patched_clean,
        'patched_corrupted_prob': patched_corr,
        'delta_clean': patched_clean - baseline_clean,
        'delta_corrupted': patched_corr - baseline_corr,
        'baseline_corr_minus_clean': baseline_corr - baseline_clean,
        'patched_corr_minus_clean': patched_corr - patched_clean,
        'delta_corr_minus_clean': (patched_corr - patched_clean) - (baseline_corr - baseline_clean),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment #10 (nnterp-style hooks)")
    parser.add_argument('--pair', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6b')
    parser.add_argument('--location', type=str, default='post_mlp', choices=['pre_attn','post_attn','post_mlp'])
    parser.add_argument('--layer', type=int, default=None, help='If set, only run this layer (0-index)')
    parser.add_argument('--max-input-tokens', type=int, default=512)
    parser.add_argument('--metric', type=str, default='counts', choices=['counts','probs'], help='Scoring metric: keyword counts in generated continuation or next-token probs')
    parser.add_argument('--gen-tokens', type=int, default=48, help='Tokens to generate for counts metric')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature for counts metric')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p for sampling when temperature>0')
    parser.add_argument('--device', type=str, default=None, choices=['cpu','cuda'])
    parser.add_argument('--samples', type=int, default=1, help='Number of stochastic samples to average (set temperature>0)')
    parser.add_argument('--dump-texts', action='store_true', help='Dump baseline/patched continuations for each layer')
    args = parser.parse_args()

    config = MATCHED_PAIRS[args.pair - 1]

    print("=" * 80)
    print(f"EXPERIMENT #10: nnterp-style hooks (Pair {args.pair}/4)")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Pair: {config['baseline_id']} → {config['target_id']}")
    print(f"Location: {args.location}")

    # Device / dtype
    global DEVICE
    if args.device:
        DEVICE = args.device
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32

    # Load prompts
    print("\nLoading prompts...")
    source_prompt = load_prompts(config['baseline_id'], config['target_id'], 'no_more_thinking')
    target_prompt = load_prompts(config['baseline_id'], config['target_id'], 'allow_more_thinking')
    print(f"Source: {len(source_prompt)} chars")
    print(f"Target: {len(target_prompt)} chars")

    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("✓ Model loaded")
    try:
        n_layers = model.config.num_hidden_layers
    except Exception:
        # Qwen naming
        n_layers = len(model.model.layers)
    print(f"  Layers: {n_layers}")

    # Encode inputs
    src_inputs = encode(tokenizer, source_prompt, DEVICE, args.max_input_tokens)
    tgt_inputs = encode(tokenizer, target_prompt, DEVICE, args.max_input_tokens)

    # Keywords
    clean_keywords = config['clean_keywords']
    corrupted_keywords = config.get('corrupted_keywords') or [config['corrupted_twist']]

    # Layer iteration
    if args.layer is not None:
        if args.layer < 0 or args.layer >= n_layers:
            print(f"Requested layer {args.layer} out of range [0,{n_layers-1}] — exiting")
            return
        layer_iter = [args.layer]
    else:
        layer_iter = list(range(n_layers))

    print("\n" + "=" * 80)
    print("SCANNING LAYERS (nnterp hooks)")
    print("=" * 80)

    results: List[Dict] = []
    samples_dir: Optional[Path] = None
    if args.dump_texts:
        samples_dir = OUTPUT_DIR / f"samples_pair{args.pair}_{args.location}"
    for L in tqdm(layer_iter, desc="Layers"):
        try:
            res = run_layer_patch(
                model, tokenizer, src_inputs, tgt_inputs,
                layer=L, location=args.location,
                clean_keywords=clean_keywords,
                corrupted_keywords=corrupted_keywords,
                metric=args.metric,
                gen_tokens=args.gen_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                samples=args.samples,
                dump_texts_dir=(samples_dir if args.dump_texts else None)
            )
            results.append(res)
        except RuntimeError as e:
            print(f"  ⚠ Layer {L} failed: {e}")
        finally:
            if DEVICE == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # Rank by override differential
    results_sorted = sorted(results, key=lambda x: x.get('delta_corr_minus_clean', 0.0), reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"pair{args.pair}_nnterp_{args.location}.json"
    with open(out_file, 'w') as f:
        json.dump({
            'pair_config': config,
            'model': args.model,
            'location': args.location,
            'max_input_tokens': args.max_input_tokens,
            'results': results_sorted,
            'top_10': results_sorted[:10]
        }, f, indent=2)

    print(f"\n✓ Results saved to: {out_file}")

    print("\n" + "=" * 80)
    print("TOP 10 LAYERS")
    print("=" * 80)
    print(f"{'Layer':<8} {'bClean':<10} {'pClean':<10} {'Δclean':<9} {'b( corr-clean )':<18} {'p( corr-clean )':<18} {'Δdiff':<10}")
    print("-" * 80)
    if not results_sorted:
        print("(no results)")
    else:
        for r in results_sorted[:10]:
            print(
                f"{r['layer']:<8} "
                f"{r.get('baseline_clean_prob', 0.0):<10.6f} "
                f"{r.get('patched_clean_prob', 0.0):<10.6f} "
                f"{r.get('delta_clean', 0.0):+9.6f} "
                f"{r.get('baseline_corr_minus_clean', 0.0):<18.6f} "
                f"{r.get('patched_corr_minus_clean', 0.0):<18.6f} "
                f"{r.get('delta_corr_minus_clean', 0.0):+10.6f}"
            )


if __name__ == "__main__":
    main()
