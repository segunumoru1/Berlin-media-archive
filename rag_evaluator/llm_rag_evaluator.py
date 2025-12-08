"""
LLM-as-Judge RAG Evaluator
Production-grade evaluation system using LLM as an evaluator.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetricType(Enum):
    """Types of evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CITATION_QUALITY = "citation_quality"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_COMPLETENESS = "answer_completeness"


@dataclass
class EvaluationResult:
    """Single metric evaluation result."""
    metric_type: MetricType
    score: float
    reason: Optional[str] = None


@dataclass
class ComprehensiveEvaluation:
    """Comprehensive evaluation result for a single response."""
    question: str
    answer: str
    retrieved_context: List[str]
    faithfulness: float
    relevance: float
    citation_quality: float
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_completeness: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score from all metrics."""
        scores = [self.faithfulness, self.relevance, self.citation_quality]
        if self.context_precision is not None:
            scores.append(self.context_precision)
        if self.context_recall is not None:
            scores.append(self.context_recall)
        if self.answer_completeness is not None:
            scores.append(self.answer_completeness)
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def grade(self) -> str:
        """Get letter grade based on overall score."""
        score = self.overall_score
        if score >= 0.9: return "A"
        if score >= 0.8: return "B"
        if score >= 0.7: return "C"
        if score >= 0.6: return "D"
        return "F"
    
    @property
    def is_production_ready(self) -> bool:
        """Check if response meets production quality standards."""
        return (self.faithfulness >= 0.8 and 
                self.relevance >= 0.8 and 
                self.citation_quality >= 0.7 and
                len(self.errors) == 0)


@dataclass
class BatchEvaluationResult:
    """Batch evaluation results."""
    total_cases: int
    individual_results: List[ComprehensiveEvaluation]
    
    @property
    def avg_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.individual_results) / len(self.individual_results)
    
    @property
    def avg_relevance(self) -> float:
        return sum(r.relevance for r in self.individual_results) / len(self.individual_results)
    
    @property
    def avg_citation_quality(self) -> float:
        return sum(r.citation_quality for r in self.individual_results) / len(self.individual_results)
    
    @property
    def avg_overall(self) -> float:
        return sum(r.overall_score for r in self.individual_results) / len(self.individual_results)
    
    @property
    def production_ready_count(self) -> int:
        return sum(1 for r in self.individual_results if r.is_production_ready)
    
    @property
    def production_ready_percentage(self) -> float:
        return (self.production_ready_count / self.total_cases * 100) if self.total_cases > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cases": self.total_cases,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevance": self.avg_relevance,
            "avg_citation_quality": self.avg_citation_quality,
            "avg_overall": self.avg_overall,
            "production_ready_count": self.production_ready_count,
            "production_ready_percentage": self.production_ready_percentage,
            "individual_results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "faithfulness": r.faithfulness,
                    "relevance": r.relevance,
                    "citation_quality": r.citation_quality,
                    "overall_score": r.overall_score,
                    "grade": r.grade,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in self.individual_results
            ]
        }


class LLMRAGEvaluator:
    """
    LLM-as-Judge evaluator for RAG system responses.
    Uses GPT-4 to evaluate faithfulness, relevance, and citation quality.
    """
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM evaluator.
        
        Args:
            model: OpenAI model to use for evaluation
            temperature: Temperature for LLM (0 for deterministic)
            api_key: OpenAI API key (or use env variable)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"LLMRAGEvaluator initialized with model: {model}")
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        retrieved_context: List[str],
        ground_truth: Optional[str] = None
    ) -> ComprehensiveEvaluation:
        """
        Comprehensive evaluation of a RAG response.
        
        Args:
            question: Original user question
            answer: Generated answer from RAG system
            retrieved_context: List of retrieved context chunks
            ground_truth: Optional ground truth answer for additional metrics
            
        Returns:
            ComprehensiveEvaluation object with all metrics
        """
        try:
            logger.info("Starting comprehensive evaluation")
            
            errors = []
            warnings = []
            
            # Core metrics (required)
            faithfulness = self._evaluate_faithfulness(answer, retrieved_context)
            logger.info(f"Faithfulness score: {faithfulness:.3f}")
            
            relevance = self._evaluate_relevance(question, answer)
            logger.info(f"Relevance score: {relevance:.3f}")
            
            citation_quality = self._evaluate_citation_quality(answer, retrieved_context)
            logger.info(f"Citation quality score: {citation_quality:.3f}")
            
            # Optional metrics
            context_precision = None
            context_recall = None
            answer_completeness = None
            
            if ground_truth:
                context_precision = self._evaluate_context_precision(
                    question, retrieved_context, ground_truth
                )
                context_recall = self._evaluate_context_recall(
                    question, retrieved_context, ground_truth
                )
                answer_completeness = self._evaluate_answer_completeness(
                    answer, ground_truth
                )
            
            # Check for issues
            if faithfulness < 0.7:
                errors.append(f"Low faithfulness score: {faithfulness:.2f} - possible hallucinations")
            
            if relevance < 0.7:
                errors.append(f"Low relevance score: {relevance:.2f} - answer may not address question")
            
            if citation_quality < 0.5:
                warnings.append(f"Poor citation quality: {citation_quality:.2f}")
            
            # Create comprehensive evaluation
            evaluation = ComprehensiveEvaluation(
                question=question,
                answer=answer,
                retrieved_context=retrieved_context,
                faithfulness=faithfulness,
                relevance=relevance,
                citation_quality=citation_quality,
                context_precision=context_precision,
                context_recall=context_recall,
                answer_completeness=answer_completeness,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Evaluation complete. Overall score: {evaluation.overall_score:.3f}, Grade: {evaluation.grade}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            
            # Return failed evaluation
            return ComprehensiveEvaluation(
                question=question,
                answer=answer,
                retrieved_context=retrieved_context,
                faithfulness=0.0,
                relevance=0.0,
                citation_quality=0.0,
                errors=[f"Evaluation failed: {str(e)}"]
            )
    
    def _evaluate_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Evaluate faithfulness (no hallucinations).
        All claims in answer must be supported by context.
        
        Args:
            answer: Generated answer
            context: Retrieved context chunks
            
        Returns:
            Faithfulness score (0-1)
        """
        try:
            context_text = "\n\n---\n\n".join(f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(context))
            
            prompt = f"""You are an expert fact-checker evaluating a RAG system's answer for faithfulness.

Your task: Determine if the answer is FULLY SUPPORTED by the provided context, with NO hallucinations or unsupported claims.

CONTEXT:
{context_text}

ANSWER TO EVALUATE:
{answer}

EVALUATION CRITERIA:
- Score 1.0: Every claim in the answer is directly supported by the context. No invented information.
- Score 0.9: Answer is faithful with only trivial connecting words that don't change meaning.
- Score 0.7-0.8: One minor unsupported detail, but main claims are correct.
- Score 0.5-0.6: Some claims supported, some not. Mixed accuracy.
- Score 0.3-0.4: Significant unsupported claims or misinterpretations.
- Score 0.0-0.2: Major hallucinations, answer invents information not in context.

IMPORTANT:
- Be strict: If the answer makes claims not in the context, lower the score.
- Numbers, dates, names must be exact.
- Inferences are acceptable only if clearly supported by context.

Provide your response in this exact format:
SCORE: [a single number between 0.0 and 1.0]
REASON: [one sentence explaining your score]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse score
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.0
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate relevance - does the answer address the question?
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevance score (0-1)
        """
        try:
            prompt = f"""You are an expert evaluator assessing if an answer properly addresses a question.

QUESTION:
{question}

ANSWER:
{answer}

EVALUATION CRITERIA:
- Score 1.0: Answer directly and completely addresses all aspects of the question.
- Score 0.8-0.9: Answer addresses the main question well, minor aspects may be incomplete.
- Score 0.6-0.7: Answer is partially relevant but misses key aspects.
- Score 0.4-0.5: Answer is tangentially related but doesn't really answer the question.
- Score 0.0-0.3: Answer is off-topic or doesn't address the question.

Provide your response in this exact format:
SCORE: [a single number between 0.0 and 1.0]
REASON: [one sentence explaining your score]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return 0.0
    
    def _evaluate_citation_quality(self, answer: str, context: List[str]) -> float:
        """
        Evaluate citation quality - are sources properly cited?
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Citation quality score (0-1)
        """
        try:
            # Check if citations are present
            has_source_markers = any(marker in answer for marker in ["[Source:", "[Page", "[Timestamp", "(Source:", "(Page", "(Timestamp"])
            
            if not has_source_markers:
                logger.warning("No citation markers found in answer")
                return 0.0
            
            context_text = "\n\n".join(f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(context))
            
            prompt = f"""You are an expert evaluator assessing citation quality in a RAG system's answer.

AVAILABLE CONTEXT SOURCES ({len(context)} total):
{context_text}

ANSWER WITH CITATIONS:
{answer}

EVALUATION CRITERIA:
- Score 1.0: All important claims are cited with specific, accurate references (timestamps/pages). Citations are precise.
- Score 0.8-0.9: Most claims cited well, minor citation gaps.
- Score 0.6-0.7: Some citations present but incomplete or imprecise.
- Score 0.4-0.5: Few citations, or citations are vague/incorrect.
- Score 0.0-0.3: No meaningful citations or completely incorrect references.

IMPORTANT:
- Citations should be specific (e.g., "04:23" or "Page 5", not just "Source 1")
- Each major claim should have a citation
- Citations must match the available sources

Provide your response in this exact format:
SCORE: [a single number between 0.0 and 1.0]
REASON: [one sentence explaining your score]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Citation quality evaluation failed: {e}")
            return 0.0
    
    def _evaluate_context_precision(
        self,
        question: str,
        context: List[str],
        ground_truth: str
    ) -> float:
        """
        Evaluate context precision - are retrieved contexts relevant?
        
        Args:
            question: Original question
            context: Retrieved context chunks
            ground_truth: Ground truth answer
            
        Returns:
            Context precision score (0-1)
        """
        try:
            context_text = "\n\n".join(f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(context))
            
            prompt = f"""You are evaluating the relevance of retrieved context chunks for a question.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXTS:
{context_text}

TASK: How many of the retrieved contexts are actually relevant to answering the question?

SCORING:
- Score = (Number of relevant contexts) / (Total contexts)
- A context is relevant if it helps answer the question or supports the ground truth

Provide your response in this exact format:
RELEVANT: [number of relevant contexts]
TOTAL: {len(context)}
SCORE: [calculated score between 0.0 and 1.0]
REASON: [brief explanation]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            return 0.0
    
    def _evaluate_context_recall(
        self,
        question: str,
        context: List[str],
        ground_truth: str
    ) -> float:
        """
        Evaluate context recall - did we retrieve all necessary information?
        
        Args:
            question: Original question
            context: Retrieved context chunks
            ground_truth: Ground truth answer
            
        Returns:
            Context recall score (0-1)
        """
        try:
            context_text = "\n\n".join(context)
            
            prompt = f"""You are evaluating if the retrieved context contains all information needed to answer the question.

QUESTION:
{question}

GROUND TRUTH ANSWER (what should be answered):
{ground_truth}

RETRIEVED CONTEXT:
{context_text}

TASK: Does the context contain all the key information needed to produce the ground truth answer?

SCORING:
- Score 1.0: All key information from ground truth is present in context
- Score 0.7-0.9: Most information present, minor details missing
- Score 0.4-0.6: Some key information missing
- Score 0.0-0.3: Major information gaps

Provide your response in this exact format:
SCORE: [a single number between 0.0 and 1.0]
REASON: [one sentence explaining your score]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Context recall evaluation failed: {e}")
            return 0.0
    
    def _evaluate_answer_completeness(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate answer completeness compared to ground truth.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Completeness score (0-1)
        """
        try:
            prompt = f"""Compare the generated answer to the ground truth answer.

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{answer}

TASK: How complete is the generated answer compared to ground truth?

SCORING:
- Score 1.0: Generated answer covers all key points from ground truth
- Score 0.7-0.9: Most key points covered, minor omissions
- Score 0.4-0.6: Some key points covered, significant omissions
- Score 0.0-0.3: Major information missing from generated answer

Provide your response in this exact format:
SCORE: [a single number between 0.0 and 1.0]
REASON: [one sentence explaining your score]"""

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            score = self._parse_score_from_response(result_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Answer completeness evaluation failed: {e}")
            return 0.0
    
    def _parse_score_from_response(self, response_text: str) -> float:
        """
        Parse score from LLM response.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Parsed score
        """
        try:
            # Look for "SCORE: X.X" pattern
            if "SCORE:" in response_text:
                score_line = [line for line in response_text.split("\n") if "SCORE:" in line][0]
                score_str = score_line.split("SCORE:")[1].strip().split()[0]
                return float(score_str)
            
            # Fallback: try to extract first float
            import re
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                return float(numbers[0])
            
            logger.warning(f"Could not parse score from: {response_text}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Score parsing failed: {e}")
            return 0.0
    
    def batch_evaluate(
        self,
        test_cases: List[Dict],
        save_results: bool = True,
        output_path: Optional[str] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple test cases in batch.
        
        Args:
            test_cases: List of dicts with 'question', 'answer', 'context', optional 'ground_truth'
            save_results: Whether to save results to file
            output_path: Path to save results (default: ./output/evaluation_results.json)
            
        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")
        
        individual_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}")
            
            result = self.evaluate_response(
                question=test_case["question"],
                answer=test_case["answer"],
                retrieved_context=test_case["context"],
                ground_truth=test_case.get("ground_truth")
            )
            
            individual_results.append(result)
        
        # Create batch result
        batch_result = BatchEvaluationResult(
            total_cases=len(test_cases),
            individual_results=individual_results
        )
        
        # Save results if requested
        if save_results:
            output_path = output_path or "./output/evaluation_results.json"
            self._save_results(batch_result, output_path)
        
        logger.info(f"Batch evaluation complete. Average score: {batch_result.avg_overall:.3f}")
        logger.info(f"Production-ready: {batch_result.production_ready_count}/{batch_result.total_cases} ({batch_result.production_ready_percentage:.1f}%)")
        
        return batch_result
    
    def _save_results(self, batch_result: BatchEvaluationResult, output_path: str):
        """Save batch results to JSON file."""
        try:
            from pathlib import Path
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_result.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")