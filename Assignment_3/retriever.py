"""Embedding model + FAISS vector store (exact inner-product search on normalized vectors = cosine)."""
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from tracing import sync


class Retriever:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = SentenceTransformer(cfg.embedding_model, device=device)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length
        # [CLS] and [SEP] take two positions, so a chunk can hold at most max_seq_length - 2 tokens
        # without being truncated when it is embedded.
        self.effective_chunk_size = min(cfg.chunk_size, self.max_seq_length - 2)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.dim = (self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension")
                    else self.model.get_sentence_embedding_dimension())
        self.index = None
        self.chunks = []

    def build(self, chunks, batch_size=32):
        self.chunks = chunks
        emb = self.model.encode([c.text for c in chunks], batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return {"n_chunks": len(chunks), "embedding_dim": int(emb.shape[1]),
                "index_bytes": int(emb.nbytes), "tokens_embedded": int(sum(c.n_tokens for c in chunks))}

    def retrieve(self, question):
        """Returns retrieved chunks plus a timing breakdown (query encoding vs. vector search)."""
        query = self.cfg.query_prefix + question
        sync(self.device)
        t0 = time.perf_counter()
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)
        sync(self.device)
        t1 = time.perf_counter()
        scores, ids = self.index.search(q, self.cfg.top_k)
        t2 = time.perf_counter()
        hits = [{"rank": r + 1, "chunk_id": self.chunks[i].chunk_id, "score": float(s),
                 "doc_id": self.chunks[i].doc_id, "page_start": self.chunks[i].page_start,
                 "page_end": self.chunks[i].page_end, "embed_tokens": self.chunks[i].n_tokens,
                 "text": self.chunks[i].text}
                for r, (i, s) in enumerate(zip(ids[0], scores[0]))]
        query_tokens = len(self.tokenizer(query, add_special_tokens=True)["input_ids"])
        return {"query": query, "query_embed_tokens": query_tokens, "hits": hits,
                "encode_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000,
                "retrieval_ms": (t2 - t0) * 1000}
