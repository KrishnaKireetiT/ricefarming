"""
ColBERT Retrieval — lightonai/Reason-ModernColBERT

No pylate. No sentence-transformers. Uses only:
  transformers, huggingface_hub, safetensors, torch, numpy

Provides:
    encode(texts, is_query=False) -> list[np.ndarray]   # (tokens, 128) per text
    maxsim_score(query_emb, doc_emb) -> float
"""

import os
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("colbert_retrieval")

MODEL_ID   = "lightonai/Reason-ModernColBERT"
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Model (lazy singleton) ─────────────────────────────────────────────────

_tokenizer = None
_backbone  = None
_dense     = None


def get_model():
    global _tokenizer, _backbone, _dense
    if _tokenizer is not None:
        return _tokenizer, _backbone, _dense

    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    logger.info("Loading ColBERT tokenizer: %s ...", MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("Loading ColBERT backbone ...")
    _backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

    # The model has a sentence-transformers Dense projection layer (hidden → 128)
    # stored separately as 1_Dense/model.safetensors
    logger.info("Loading Dense projection layer ...")
    dense_cfg_path    = hf_hub_download(MODEL_ID, "1_Dense/config.json")
    dense_weight_path = hf_hub_download(MODEL_ID, "1_Dense/model.safetensors")

    with open(dense_cfg_path) as f:
        dense_cfg = json.load(f)

    in_features  = dense_cfg["in_features"]
    out_features = dense_cfg["out_features"]
    use_bias     = dense_cfg.get("bias", True)

    _dense = torch.nn.Linear(in_features, out_features, bias=use_bias).to(DEVICE).eval()
    weights = load_file(dense_weight_path)
    _dense.weight.data = weights["linear.weight"].to(DEVICE)
    if use_bias and "linear.bias" in weights:
        _dense.bias.data = weights["linear.bias"].to(DEVICE)

    logger.info("ColBERT ready — backbone hidden=%d, projection→%d, device=%s",
                in_features, out_features, DEVICE)
    return _tokenizer, _backbone, _dense


# ─── Encoding ────────────────────────────────────────────────────────────────

def encode(texts: list[str], is_query: bool = False) -> list[np.ndarray]:
    """
    Returns a list of (num_tokens, 128) float32 arrays — one per text.
    Padding tokens are removed. Vectors are L2-normalised.
    """
    tokenizer, backbone, dense = get_model()
    results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t if t and t.strip() else "empty" for t in texts[i : i + BATCH_SIZE]]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512 if is_query else 8192,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out        = backbone(**inputs)
            token_embs = out.last_hidden_state          # (B, T, hidden)
            projected  = dense(token_embs)              # (B, T, 128)
            projected  = F.normalize(projected, p=2, dim=-1)

        attn_mask     = inputs["attention_mask"].cpu().numpy()   # (B, T)
        projected_cpu = projected.cpu().numpy()

        for j in range(len(batch)):
            mask = attn_mask[j].astype(bool)
            results.append(projected_cpu[j][mask])      # (valid_tokens, 128)

    return results


# ─── MaxSim ──────────────────────────────────────────────────────────────────

def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """
    query_emb : (Q, 128)
    doc_emb   : (N, 128)
    For each query token, find its max cosine sim with any doc token, then sum.
    """
    scores = query_emb @ doc_emb.T      # (Q, N)
    return float(scores.max(axis=1).sum())
