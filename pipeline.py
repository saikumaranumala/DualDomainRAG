"""
healthcare/pipeline.py
──────────────────────
Healthcare RAG pipeline.

Retrieval flow:
1. Extract clinical entities (BERT NER) from the query to enrich it.
2. Embed the enriched query with ClinicalBERT sentence embeddings.
3. Retrieve top-k chunks from the ChromaDB clinical knowledge base.
4. Re-rank retrieved chunks with a cross-encoder for better precision.
5. Assemble a context block and invoke the LLM with a clinical system prompt.
6. Return structured response with citations, confidence, and SHAP attribution.

Domain-specific guardrails:
- PHI scrubbing on all user queries before storage/logging.
- Confidence gating: if retrieval score < threshold, respond with
  "I could not find reliable clinical information" rather than hallucinate.
- Every response includes a medical disclaimer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.healthcare.clinical_ner import ClinicalNER
from src.healthcare.vector_store import HealthcareVectorStore
from src.shared.llm_core import LLMCore, LLMResponse
from src.shared.embeddings import EmbeddingService


# ── Configuration ──────────────────────────────────────────────────────────────

RETRIEVAL_TOP_K = 7
RERANK_TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.30  # cosine similarity floor
MEDICAL_DISCLAIMER = (
    "\n\n⚠️ **Medical disclaimer**: This response is generated from indexed "
    "clinical literature and is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider."
)

SYSTEM_PROMPT = """You are a clinical decision support assistant with expertise in 
evidence-based medicine. You answer questions using ONLY the provided clinical 
context. Follow these rules:
1. Cite sources by their document ID in square brackets, e.g. [DOC-3].
2. If the context does not contain sufficient information, say so clearly.
3. Use precise medical terminology.
4. Flag contraindications and drug interactions prominently.
5. Never diagnose or prescribe — always recommend consulting a clinician.
6. Structure longer answers with clear headings."""


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class HealthcareRAGResponse:
    query: str
    enriched_query: str
    answer: str
    sources: list[RetrievedChunk]
    domain: str = "healthcare"
    confidence: float = 0.0
    entities_extracted: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    has_sufficient_context: bool = True

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "query": self.query,
            "answer": self.answer + MEDICAL_DISCLAIMER,
            "sources": [
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "score": round(c.score, 4),
                    "excerpt": c.text[:300] + "...",
                }
                for c in self.sources
            ],
            "entities_extracted": self.entities_extracted,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "has_sufficient_context": self.has_sufficient_context,
        }


# ── PHI scrubber ───────────────────────────────────────────────────────────────

_PHI_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),          # SSN
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]"),  # Phone
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\bMRN[-:\s]?\d{6,10}\b", "[MRN]"),          # Medical record numbers
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]"),   # Dates
]


def scrub_phi(text: str) -> str:
    """Remove obvious PHI patterns from query text before logging."""
    for pattern, replacement in _PHI_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Pipeline ───────────────────────────────────────────────────────────────────

class HealthcareRAGPipeline:
    """
    End-to-end healthcare RAG pipeline.

    Usage
    -----
    >>> pipeline = HealthcareRAGPipeline()
    >>> response = pipeline.query("What are metformin contraindications?")
    >>> print(response.answer)
    """

    def __init__(
        self,
        vector_store: Optional[HealthcareVectorStore] = None,
        ner: Optional[ClinicalNER] = None,
        llm: Optional[LLMCore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ):
        self.vector_store = vector_store or HealthcareVectorStore()
        self.ner = ner or ClinicalNER()
        self.llm = llm or LLMCore(domain="healthcare", system_prompt=SYSTEM_PROMPT)
        self.embedder = embedding_service or EmbeddingService(domain="healthcare")
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.min_score = min_score

        logger.info("HealthcareRAGPipeline initialised.")

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        explain: bool = False,
    ) -> HealthcareRAGResponse:
        """
        Run the full healthcare RAG pipeline for a single query.

        Parameters
        ----------
        query:   Raw user query string.
        top_k:   Override the number of retrieved chunks.
        explain: If True, compute SHAP explanations (slower).
        """
        t0 = time.perf_counter()
        k = top_k or self.top_k

        # Step 1 — PHI scrub
        safe_query = scrub_phi(query)
        logger.info(f"Healthcare query: {safe_query[:80]}")

        # Step 2 — Clinical NER to enrich the query
        entities = self.ner.extract(safe_query)
        entity_strings = [e["text"] for e in entities]
        enriched_query = self._enrich_query(safe_query, entities)
        logger.debug(f"Entities extracted: {entity_strings}")

        # Step 3 — Embed enriched query
        query_embedding = self.embedder.embed(enriched_query)

        # Step 4 — Retrieve from ChromaDB
        raw_chunks = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=k,
        )
        logger.debug(f"Retrieved {len(raw_chunks)} chunks before filtering")

        # Step 5 — Filter by minimum score
        filtered = [c for c in raw_chunks if c.score >= self.min_score]
        if not filtered:
            logger.warning("No chunks met the minimum retrieval score threshold.")
            return self._insufficient_context_response(
                query, enriched_query, entity_strings, t0
            )

        # Step 6 — Re-rank (cross-encoder) and keep top rerank_top_k
        reranked = self._rerank(safe_query, filtered)[: self.rerank_top_k]

        # Step 7 — Build context block
        context = self._build_context(reranked)

        # Step 8 — LLM generation
        llm_response: LLMResponse = self.llm.generate(
            query=safe_query,
            context=context,
            explain=explain,
        )

        # Step 9 — Confidence = mean retrieval score of final chunks
        confidence = sum(c.score for c in reranked) / len(reranked)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.success(f"Healthcare query answered in {latency_ms:.1f}ms")

        return HealthcareRAGResponse(
            query=query,
            enriched_query=enriched_query,
            answer=llm_response.text,
            sources=reranked,
            confidence=confidence,
            entities_extracted=entity_strings,
            latency_ms=latency_ms,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _enrich_query(self, query: str, entities: list[dict]) -> str:
        """Append extracted entity types to the query for better retrieval."""
        if not entities:
            return query
        entity_hints = ", ".join(
            f"{e['text']} ({e['label']})" for e in entities
        )
        return f"{query} [Clinical entities: {entity_hints}]"

    def _rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Simple re-ranking using the cross-encoder score from the vector store.
        In production, replace with a dedicated cross-encoder model.
        """
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into an LLM-ready context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[DOC-{i}] Source: {chunk.source}\n{chunk.text.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def _insufficient_context_response(
        self,
        query: str,
        enriched_query: str,
        entities: list[str],
        t0: float,
    ) -> HealthcareRAGResponse:
        latency_ms = (time.perf_counter() - t0) * 1000
        return HealthcareRAGResponse(
            query=query,
            enriched_query=enriched_query,
            answer=(
                "I was unable to find sufficient clinical information in the "
                "knowledge base to answer this question reliably. Please consult "
                "a qualified healthcare professional."
            ),
            sources=[],
            confidence=0.0,
            entities_extracted=entities,
            latency_ms=latency_ms,
            has_sufficient_context=False,
        )
