"""
retrieval.py — Lightweight semantic retrieval using sentence-transformers + numpy.

Why no ChromaDB:
- ChromaDB is overkill for a 40-entry knowledge base.
- Its dependency tree (opentelemetry, grpc, pypika, etc.) causes version
  conflicts on Streamlit Cloud.
- A simple numpy cosine-similarity search is faster to start, uses less
  memory, and has zero conflict risk.

API is identical to the previous ChromaDB version so nothing else changes.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KB_PATH = Path(__file__).parent / "knowledge_base.json"


class Retriever:
    """Wraps sentence-transformers + numpy for semantic search. Loaded once."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.entries: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load_kb()

    def _load_kb(self) -> None:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        documents = []
        for entry in raw_entries:
            # Embed question + answer + symptoms together. This helps the
            # model match on symptom descriptions, not just questions.
            symptom_text = " ".join(entry.get("symptoms", []))
            doc_text = f"{entry['question']} {entry['answer']} {symptom_text}"
            documents.append(doc_text)
            self.entries.append(entry)

        # Encode all documents once. normalize_embeddings=True means we can
        # use a simple dot product for cosine similarity later.
        embs = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.asarray(embs, dtype=np.float32)
        logger.info("Loaded %d KB entries into in-memory index", len(self.entries))

    def search(
        self,
        query: str,
        *,
        crop: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Dict]:
        """
        Search the KB. Returns a list of dicts with question, answer,
        metadata, and a similarity score in [0, 1] (higher = better).
        """
        if self.embeddings is None or len(self.entries) == 0:
            return []

        # Encode query and compute cosine similarity with all entries.
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_emb, dtype=np.float32)[0]
        # Since both are L2-normalized, dot product = cosine similarity.
        sims = self.embeddings @ query_vec  # shape: (n_entries,)

        # Build candidate list with metadata filtering.
        candidates = []
        for i, entry in enumerate(self.entries):
            # Metadata filter: crop must match (with "both" as wildcard)
            if crop and crop != "general":
                entry_crop = entry.get("crop", "unknown")
                if entry_crop not in (crop, "both"):
                    continue
            # Category filter (usually unused — we let semantics do the work)
            if category and category != "general":
                if entry.get("category") != category:
                    continue

            candidates.append({
                "idx": i,
                "score": float(sims[i]),
                "entry": entry,
            })

        # Sort by similarity descending and take top_k
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:top_k]

        hits = []
        for c in top:
            entry = c["entry"]
            hits.append({
                "id": entry.get("id", ""),
                "question": entry["question"],
                "answer": entry["answer"],
                "crop": entry.get("crop", "unknown"),
                "category": entry.get("category", "general"),
                "confidence": entry.get("confidence", "medium"),
                "source": entry.get("source", "Knowledge Base"),
                "score": c["score"],
            })
        return hits
