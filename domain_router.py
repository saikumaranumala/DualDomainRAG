"""
domain_router.py
────────────────
Zero-shot domain classifier using DeBERTa-v3-large-mnli-fever-anli-ling-wanli.
Routes every incoming query to either 'healthcare' or 'fintech' before
the appropriate RAG pipeline is invoked.

Key design decisions:
- Zero-shot so no labelled routing data is required.
- Confidence threshold (default 0.65) falls back to the 'ambiguous' class,
  which triggers a clarifying response rather than a wrong-domain answer.
- Results are cached (LRU) so repeated identical queries don't re-run inference.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from loguru import logger
from transformers import pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "cross-encoder/nli-deberta-v3-large"

HEALTHCARE_LABELS = [
    "clinical medicine or patient care",
    "pharmacy or drug treatment",
    "medical diagnosis or symptoms",
    "electronic health records or clinical notes",
    "healthcare regulations or HIPAA compliance",
]

FINTECH_LABELS = [
    "financial markets or investment",
    "banking or credit or lending",
    "fraud detection or transaction analysis",
    "SEC filings or earnings reports",
    "financial regulations or compliance",
]

CANDIDATE_LABELS = HEALTHCARE_LABELS + FINTECH_LABELS

CONFIDENCE_THRESHOLD = 0.65
CACHE_SIZE = 512


# ── Data models ───────────────────────────────────────────────────────────────

class Domain(str, Enum):
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    AMBIGUOUS = "ambiguous"


@dataclass
class RouterResult:
    domain: Domain
    confidence: float
    top_label: str
    latency_ms: float
    raw_scores: dict[str, float]

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "domain": self.domain.value,
            "confidence": round(self.confidence, 4),
            "top_label": self.top_label,
            "latency_ms": round(self.latency_ms, 2),
            "is_confident": self.is_confident,
        }


# ── Router ────────────────────────────────────────────────────────────────────

class DomainRouter:
    """
    Wraps a zero-shot NLI pipeline to classify queries into healthcare or
    fintech domains. Uses label aggregation: the winning *category* is the one
    whose constituent labels collectively score highest.

    Usage
    -----
    >>> router = DomainRouter()
    >>> result = router.classify("What are metformin side effects?")
    >>> print(result.domain)   # Domain.HEALTHCARE
    >>> print(result.confidence)  # 0.93
    """

    def __init__(
        self,
        model_name: str = CLASSIFIER_MODEL,
        device: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading domain router model: {model_name} on {_device}")
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=_device,
        )
        logger.success("Domain router ready.")

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def classify(self, query: str) -> RouterResult:
        """
        Classify a query string into a domain.
        Results are LRU-cached by query string.
        """
        t0 = time.perf_counter()

        result = self._pipe(
            query,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
        )

        # Aggregate label scores into domain scores
        label_scores = dict(zip(result["labels"], result["scores"]))
        healthcare_score = sum(label_scores[l] for l in HEALTHCARE_LABELS)
        fintech_score = sum(label_scores[l] for l in FINTECH_LABELS)

        # Normalize to [0, 1]
        total = healthcare_score + fintech_score
        h_conf = healthcare_score / total if total > 0 else 0.5
        f_conf = fintech_score / total if total > 0 else 0.5

        top_label = result["labels"][0]
        confidence = max(h_conf, f_conf)
        raw_domain = Domain.HEALTHCARE if h_conf > f_conf else Domain.FINTECH

        domain = raw_domain if confidence >= self.confidence_threshold else Domain.AMBIGUOUS

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Router: '{query[:60]}...' → {domain.value} "
            f"(conf={confidence:.3f}, {latency_ms:.1f}ms)"
        )

        return RouterResult(
            domain=domain,
            confidence=confidence,
            top_label=top_label,
            latency_ms=latency_ms,
            raw_scores={"healthcare": h_conf, "fintech": f_conf},
        )

    def classify_batch(self, queries: list[str]) -> list[RouterResult]:
        """Classify multiple queries. Cache hits are free."""
        return [self.classify(q) for q in queries]


# ── Singleton accessor ─────────────────────────────────────────────────────────

_router_instance: Optional[DomainRouter] = None


def get_router() -> DomainRouter:
    """Return (and lazily initialise) the global router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = DomainRouter()
    return _router_instance


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    router = DomainRouter()
    test_queries = [
        "What are the contraindications of metformin for Type 2 diabetes?",
        "Analyse Apple's Q3 2024 revenue growth trajectory.",
        "Is paracetamol safe during pregnancy?",
        "How do I detect credit card fraud in real-time transaction streams?",
        "What is the capital of France?",  # ambiguous / off-domain
    ]
    print(f"\n{'Query':<55} {'Domain':<12} {'Confidence'}")
    print("-" * 80)
    for q in test_queries:
        r = router.classify(q)
        print(f"{q[:54]:<55} {r.domain.value:<12} {r.confidence:.3f}")
