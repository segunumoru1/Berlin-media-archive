"""
RAG Evaluator Module
Comprehensive evaluation system for RAG responses.
"""

from .metrics import (
    ComprehensiveEvaluation,
    BatchEvaluationResult,
    EvaluationResult,
    MetricType
)

from .llm_rag_evaluator import LLMRAGEvaluator

from .test_cases import EvaluationTestCases, TestCase

try:
    from .ragas_integration import RAGASEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

__all__ = [
    "ComprehensiveEvaluation",
    "BatchEvaluationResult",
    "EvaluationResult",
    "MetricType",
    "LLMRAGEvaluator",
    "EvaluationTestCases",
    "TestCase",
]

if RAGAS_AVAILABLE:
    __all__.append("RAGASEvaluator")