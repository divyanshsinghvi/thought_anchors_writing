import json
import numpy as np
import os
from pathlib import Path
import torch
from typing import List
import argparse

from sentence_transformers import SentenceTransformer

from config import EMBEDING_MODEL
from tqdm import tqdm
import pickle

def get_paragraphs(text: str) -> List[str]:
    """
    Get paragraphs in text, treating double newlines as paragraph separators.
    """
    if not text:
        return []

    # Split by double newlines (common paragraph separator)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="Compute and save embeddings with structure-aware output directory")
    parser.add_argument("--base-dir", required=True, type=str, help="Base directory containing paragraph subdirectories")
    parser.add_argument("--out-dir", required=True, type=str, help="Base output directory where results will be written")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found or not a directory: {base_dir}")

    # Fixed mode directory, appended to output path as requested
    mode_dir = "remove_after/allow_more_thinking/"

    out_dir = Path(args.out_dir) / mode_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        EMBEDING_MODEL,
        model_kwargs={"device_map": "auto", "torch_dtype" : torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    para_names = os.listdir(base_dir)
    para_names.sort()

    # parameters
    num_paras = len(para_names)
    emb_size = model.get_sentence_embedding_dimension()

    reasoning_embs = []
    content_embs   = []
    og_embs        = []
    reasoning_masks = []   

    for para_idx, para_name in enumerate(tqdm(para_names, desc="Paragraphs", position=0)):
        curr_dir = base_dir / para_name / mode_dir
        rollout_files = sorted(
            f for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))
        )

        content_texts   = []
        reasoning_texts = []
        reasoning_mask  = [] 

        for rollout_idx, fname in enumerate(rollout_files):
            with open(curr_dir / fname, 'r') as f:
                temp = json.load(f)

                # Safely get paragraph reasoning
                paras = get_paragraphs(temp['rollout_data']['reasoning'])
                reasoning_text = paras[0] if paras else ""
                
                reasoning_texts.append(reasoning_text)
                reasoning_mask.append(1 if reasoning_text.strip() else 0)  # <--- mark empty or not

                content_texts.append(temp['rollout_data']['content'])

                # original reasoning embedding (once per paragraph)
                if rollout_idx == 0:
                    original_paras = get_paragraphs(temp['original_reasoning'])
                    if para_idx < len(original_paras):
                        orig_para = original_paras[para_idx]
                    else:
                        orig_para = ""
                    og_emb = model.encode(orig_para, convert_to_numpy=True, normalize_embeddings=True)
                    og_embs.append(og_emb)

        # batch encode
        content_batch   = model.encode(content_texts,   convert_to_numpy=True, normalize_embeddings=True, batch_size=16)
        reasoning_batch = model.encode(reasoning_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=16)

        # apply mask: zero out embeddings where reasoning text was empty
        reasoning_batch = np.array(reasoning_batch)
        mask_array = np.array(reasoning_mask).reshape(-1, 1)
        reasoning_batch = reasoning_batch * mask_array  # zero out empty reasoning embeddings

        # store paragraph-level data
        content_embs.append(content_batch)
        reasoning_embs.append(reasoning_batch)
        reasoning_masks.append(mask_array)  # <--- store mask

    # convert to arrays-of-arrays
    content_embs   = [np.array(c) for c in content_embs]
    reasoning_embs = [np.array(r) for r in reasoning_embs]
    reasoning_masks = [np.array(m) for m in reasoning_masks]
    og_embs        = np.array(og_embs)

    with open(out_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(
            {
                "content_embs": content_embs,
                "reasoning_embs": reasoning_embs,
                "reasoning_masks": reasoning_masks,
                "og_embs": og_embs,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    main()
