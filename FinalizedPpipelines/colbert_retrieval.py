"""
ColBERT Retrieval — lightonai/Reason-ModernColBERT

No pylate. No sentence-transformers. Uses only:
  transformers, huggingface_hub, safetensors, torch, numpy

Setup:  (all of these are already pulled in by transformers)
    pip install transformers huggingface_hub safetensors torch numpy

Usage:
    python colbert_retrieval.py               # interactive query loop
    python colbert_retrieval.py --rebuild     # force re-index
    python colbert_retrieval.py --query "water management for rice"
    python colbert_retrieval.py --query "..." --k 10
"""

import os
import json
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F

CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "phase1_chunks_vi.json")
EMBED_DIR   = os.path.join(os.path.dirname(__file__), "colbert-embeddings")
IDS_FILE    = os.path.join(EMBED_DIR, "chunk_ids.pkl")
MODEL_ID    = "lightonai/Reason-ModernColBERT"
BATCH_SIZE  = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Chunks ───────────────────────────────────────────────────────────────────

def load_chunks() -> dict[str, dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {str(c["chunk_id"]): c for c in chunks}


# ─── Model ────────────────────────────────────────────────────────────────────

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

    print(f"Loading tokenizer: {MODEL_ID} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading backbone ...")
    _backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

    # The model has a sentence-transformers Dense projection layer (hidden → 128)
    # stored separately as 1_Dense/model.safetensors
    print("Loading Dense projection layer ...")
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

    print(f"  Ready — backbone hidden={in_features}, projection→{out_features}, device={DEVICE}\n")
    return _tokenizer, _backbone, _dense


# ─── Encoding ─────────────────────────────────────────────────────────────────

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


# ─── Build index ──────────────────────────────────────────────────────────────

def build_index(chunks: dict[str, dict], rebuild: bool = False):
    from tqdm import tqdm

    if os.path.exists(IDS_FILE) and not rebuild:
        print(f"Embeddings already exist at '{EMBED_DIR}'. Use --rebuild to re-encode.")
        return

    os.makedirs(EMBED_DIR, exist_ok=True)

    ids   = list(chunks.keys())
    texts = [chunks[cid].get("eng_summary", "") or "empty" for cid in ids]

    print(f"Encoding {len(ids)} chunks ...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Encoding", unit="batch"):
        batch_ids   = ids[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        embs        = encode(batch_texts, is_query=False)
        for cid, emb in zip(batch_ids, embs):
            np.save(os.path.join(EMBED_DIR, f"{cid}.npy"), emb.astype(np.float32))

    with open(IDS_FILE, "wb") as f:
        pickle.dump(ids, f)

    print(f"Done — {len(ids)} chunk embeddings saved to '{EMBED_DIR}'")


# ─── MaxSim ───────────────────────────────────────────────────────────────────

def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """
    query_emb : (Q, 128)
    doc_emb   : (N, 128)
    For each query token, find its max cosine sim with any doc token, then sum.
    """
    scores = query_emb @ doc_emb.T      # (Q, N)
    return float(scores.max(axis=1).sum())


# ─── Retrieve ─────────────────────────────────────────────────────────────────

def retrieve(query: str, chunks: dict[str, dict], k: int = 5) -> list[dict]:
    query_emb = encode([query], is_query=True)[0]   # (Q, 128)

    with open(IDS_FILE, "rb") as f:
        ids = pickle.load(f)

    scored = []
    for cid in ids:
        path = os.path.join(EMBED_DIR, f"{cid}.npy")
        if not os.path.exists(path):
            continue
        doc_emb = np.load(path)
        scored.append((cid, maxsim_score(query_emb, doc_emb)))

    scored.sort(key=lambda x: x[1], reverse=True)

    hits = []
    for cid, score in scored[:k]:
        chunk = chunks.get(str(cid), {})
        hits.append({
            "chunk_id":  cid,
            "score":     round(score, 4),
            "title":     chunk.get("eng_chunk_title", ""),
            "hierarchy": chunk.get("eng_chunk_hierarchy", ""),
            "pages":     f"{chunk.get('start_page', '?')}–{chunk.get('end_page', '?')}",
            "summary":   chunk.get("eng_summary", "")[:300],
        })
    return hits


# ─── Display ──────────────────────────────────────────────────────────────────

def print_results(query: str, hits: list[dict]):
    print(f"\nQuery: \"{query}\"")
    print("─" * 60)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] chunk_id={h['chunk_id']}  score={h['score']}  pages={h['pages']}")
        print(f"    {h['title']}")
        print(f"    {h['hierarchy']}")
        print(f"    {h['summary']}...")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force re-encode even if embeddings exist")
    parser.add_argument("--query",   type=str,            help="Run a single query and exit")
    parser.add_argument("--k",       type=int, default=5, help="Number of results (default 5)")
    args = parser.parse_args()

    chunks = load_chunks()
    build_index(chunks, rebuild=args.rebuild)

    if args.query:
        hits = retrieve(args.query, chunks, k=args.k)
        print_results(args.query, hits)
        return

    print("ColBERT retrieval ready. Type a query (or 'quit' to exit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        hits = retrieve(query, chunks, k=args.k)
        print_results(query, hits)


if __name__ == "__main__":
    main()