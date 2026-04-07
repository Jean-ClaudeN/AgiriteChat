"""
retrieval.py — Semantic retrieval over the knowledge base.

Why this replaces the old keyword scoring:
- Handles paraphrases: "my corn stalks are shriveling" now matches drought
  entries even though no words overlap.
- Supports metadata filters (crop, category) so general questions don't
  leak across crops.
- Returns similarity scores we can threshold on for the confidence gate.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "agiritechat_kb"
KB_PATH = Path(__file__).parent / "knowledge_base.json"


class Retriever:
    """Wraps ChromaDB + sentence-transformers. Loaded once at startup."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        # In-memory Chroma is fine for a ~100-entry KB. No persistence needed.
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._load_kb()

    def _load_kb(self) -> None:
        if self.collection.count() > 0:
            return  # Already loaded this session

        with open(KB_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)

        ids, documents, metadatas = [], [], []
        for entry in entries:
            # Embed question + answer + symptoms together. This helps
            # the model match on symptom descriptions, not just questions.
            symptom_text = " ".join(entry.get("symptoms", []))
            doc_text = f"{entry['question']} {entry['answer']} {symptom_text}"

            ids.append(entry["id"])
            documents.append(doc_text)
            metadatas.append({
                "crop": entry.get("crop", "unknown"),
                "category": entry.get("category", "general"),
                "question": entry["question"],
                "answer": entry["answer"],
                "confidence": entry.get("confidence", "medium"),
            })

        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Loaded %d KB entries into Chroma", len(ids))

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
        query_emb = self.model.encode([query], normalize_embeddings=True).tolist()

        # Build metadata filter. Chroma uses $or/$eq syntax.
        where_clause = None
        if crop and crop != "general":
            # "both" entries should match any crop query
            where_clause = {"crop": {"$in": [crop, "both"]}}
        if category and category != "general":
            cat_filter = {"category": {"$eq": category}}
            where_clause = (
                {"$and": [where_clause, cat_filter]} if where_clause else cat_filter
            )

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=where_clause,
        )

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, _id in enumerate(results["ids"][0]):
            # Chroma returns squared cosine distance in [0, 2]. Convert to
            # similarity in [0, 1] where 1 = identical.
            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance / 2.0)
            meta = results["metadatas"][0][i]
            hits.append({
                "id": _id,
                "question": meta["question"],
                "answer": meta["answer"],
                "crop": meta["crop"],
                "category": meta["category"],
                "confidence": meta["confidence"],
                "score": similarity,
            })
        return hits
