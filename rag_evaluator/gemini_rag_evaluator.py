"""
RAG Evaluation using Google Gemini as Judge
Evaluates retrieval quality, answer quality, and citation accuracy.
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")


@dataclass
class EvaluationMetrics:
    """Metrics for RAG evaluation."""
    # Retrieval Quality
    retrieval_precision: float  # 0-1: Relevance of retrieved chunks
    retrieval_recall: float  # 0-1: Coverage of necessary information
    
    # Answer Quality
    answer_relevance: float  # 0-1: How well answer addresses the question
    answer_correctness: float  # 0-1: Factual accuracy based on context
    answer_completeness: float  # 0-1: Completeness of the answer
    
    # Citation Quality
    citation_accuracy: float  # 0-1: Accuracy of citations
    citation_coverage: float  # 0-1: How well citations support claims
    
    # Overall
    overall_score: float  # 0-1: Weighted average
    
    # Detailed Feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    query: str
    answer: str
    context_used: List[str]
    citations: List[Dict[str, Any]]
    metrics: EvaluationMetrics
    evaluator_reasoning: str
    timestamp: str
    
    def to_dict(self):
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        return result


class GeminiRAGEvaluator:
    """
    RAG Evaluator using Google Gemini as an LLM judge.
    
    Evaluates:
    1. Retrieval Quality - Are retrieved chunks relevant?
    2. Answer Quality - Is the answer accurate and complete?
    3. Citation Quality - Are citations accurate and well-used?
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-preview",
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini RAG Evaluator.
        
        Args:
            model_name: Gemini model to use (gemini-1.5-flash-preview recommended)
            api_key: Gemini API key (or from GEMINI_API_KEY env var)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get your API key from https://aistudio.google.com/app/apikey"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"Gemini RAG Evaluator initialized with model: {model_name}")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        citations: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.
        
        Args:
            query: User's question
            answer: System's answer
            retrieved_contexts: Retrieved context chunks
            citations: Citation metadata
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            EvaluationResult with detailed metrics
        """
        logger.info(f"Evaluating query: {query[:50]}...")
        
        try:
            # Build evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                query=query,
                answer=answer,
                contexts=retrieved_contexts,
                citations=citations,
                ground_truth=ground_truth
            )
            
            # Get evaluation from Gemini
            logger.debug("Sending evaluation request to Gemini...")
            response = self.model.generate_content(evaluation_prompt)
            evaluation_text = response.text
            
            logger.debug(f"Gemini evaluation response: {evaluation_text[:200]}...")
            
            # Parse evaluation
            metrics = self._parse_evaluation(evaluation_text)
            
            # Create result
            from datetime import datetime
            result = EvaluationResult(
                query=query,
                answer=answer,
                context_used=retrieved_contexts,
                citations=citations,
                metrics=metrics,
                evaluator_reasoning=evaluation_text,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Evaluation complete. Overall score: {metrics.overall_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
    
    def _build_evaluation_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        citations: List[Dict[str, Any]],
        ground_truth: Optional[str]
    ) -> str:
        """Build the evaluation prompt for Gemini."""
        
        # Format contexts
        contexts_text = "\n\n---\n\n".join([
            f"Context {i+1}:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # Format citations
        citations_text = "\n".join([
            f"Citation {i+1}: {cite.get('source_file', 'Unknown')} - "
            f"{'Speaker: ' + cite.get('speaker', 'N/A') if cite.get('source_type') == 'audio' else 'Page: ' + str(cite.get('page_number', 'N/A'))}"
            for i, cite in enumerate(citations)
        ])
        
        prompt = f"""You are an expert RAG (Retrieval-Augmented Generation) system evaluator. Your task is to evaluate the quality of a question-answering system's response.

## INPUT

**User Question:**
{query}

**Retrieved Contexts:**
{contexts_text}

**System Answer:**
{answer}

**Citations Used:**
{citations_text}

{"**Ground Truth Answer (for reference):**" if ground_truth else ""}
{ground_truth if ground_truth else ""}

---

## EVALUATION CRITERIA

Evaluate the system on these dimensions (score 0.0 to 1.0):

### 1. RETRIEVAL QUALITY
- **Retrieval Precision**: Are the retrieved contexts relevant to the question?
- **Retrieval Recall**: Do the contexts contain all necessary information to answer the question?

### 2. ANSWER QUALITY
- **Answer Relevance**: Does the answer directly address the question?
- **Answer Correctness**: Is the answer factually correct based on the provided contexts?
- **Answer Completeness**: Does the answer cover all important aspects of the question?

### 3. CITATION QUALITY
- **Citation Accuracy**: Are the citations correctly attributed to their sources?
- **Citation Coverage**: Do the citations adequately support the claims made in the answer?

---

## OUTPUT FORMAT

Provide your evaluation in this EXACT format:

RETRIEVAL_PRECISION: [score 0.0-1.0]
RETRIEVAL_RECALL: [score 0.0-1.0]
ANSWER_RELEVANCE: [score 0.0-1.0]
ANSWER_CORRECTNESS: [score 0.0-1.0]
ANSWER_COMPLETENESS: [score 0.0-1.0]
CITATION_ACCURACY: [score 0.0-1.0]
CITATION_COVERAGE: [score 0.0-1.0]

STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

REASONING:
[Detailed explanation of your evaluation, including specific examples from the answer and contexts]

---

Begin your evaluation:"""
        
        return prompt
    
    def _parse_evaluation(self, evaluation_text: str) -> EvaluationMetrics:
        """Parse Gemini's evaluation response into structured metrics."""
        
        import re
        
        # Extract scores using regex
        def extract_score(pattern: str, text: str, default: float = 0.5) -> float:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # Clamp to 0-1
                except ValueError:
                    logger.warning(f"Could not parse score for {pattern}")
                    return default
            return default
        
        # Extract scores
        retrieval_precision = extract_score(r'RETRIEVAL_PRECISION:\s*([\d.]+)', evaluation_text)
        retrieval_recall = extract_score(r'RETRIEVAL_RECALL:\s*([\d.]+)', evaluation_text)
        answer_relevance = extract_score(r'ANSWER_RELEVANCE:\s*([\d.]+)', evaluation_text)
        answer_correctness = extract_score(r'ANSWER_CORRECTNESS:\s*([\d.]+)', evaluation_text)
        answer_completeness = extract_score(r'ANSWER_COMPLETENESS:\s*([\d.]+)', evaluation_text)
        citation_accuracy = extract_score(r'CITATION_ACCURACY:\s*([\d.]+)', evaluation_text)
        citation_coverage = extract_score(r'CITATION_COVERAGE:\s*([\d.]+)', evaluation_text)
        
        # Calculate overall score (weighted average)
        overall_score = (
            retrieval_precision * 0.15 +
            retrieval_recall * 0.15 +
            answer_relevance * 0.20 +
            answer_correctness * 0.20 +
            answer_completeness * 0.15 +
            citation_accuracy * 0.075 +
            citation_coverage * 0.075
        )
        
        # Extract lists - FIX: Use raw string
        def extract_list(section_name: str, text: str) -> List[str]:
            # Fix: Use raw string (r'...') for regex patterns
            pattern = rf'{section_name}:\s*\n((?:- .+\n?)+)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                items_text = match.group(1)
                items = re.findall(r'- (.+)', items_text)
                return [item.strip() for item in items if item.strip()]
            return []
        
        strengths = extract_list('STRENGTHS', evaluation_text)
        weaknesses = extract_list('WEAKNESSES', evaluation_text)
        suggestions = extract_list('SUGGESTIONS', evaluation_text)
        
        # Ensure at least some feedback
        if not strengths:
            strengths = ["Evaluation completed"]
        if not weaknesses:
            weaknesses = ["No significant weaknesses identified"]
        if not suggestions:
            suggestions = ["Continue monitoring quality"]
        
        return EvaluationMetrics(
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            answer_relevance=answer_relevance,
            answer_correctness=answer_correctness,
            answer_completeness=answer_completeness,
            citation_accuracy=citation_accuracy,
            citation_coverage=citation_coverage,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases with query, answer, contexts, citations
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Evaluating batch of {len(test_cases)} test cases...")
        
        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}...")
            
            try:
                result = self.evaluate(
                    query=test_case['query'],
                    answer=test_case['answer'],
                    retrieved_contexts=test_case['contexts'],
                    citations=test_case.get('citations', []),
                    ground_truth=test_case.get('ground_truth')
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate test case {i+1}: {e}")
                continue
        
        logger.info(f"✅ Batch evaluation complete: {len(results)}/{len(test_cases)} successful")
        
        return results
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate evaluation report with statistics.
        
        Args:
            results: List of evaluation results
            output_path: Optional path to save JSON report
            
        Returns:
            Report dictionary
        """
        if not results:
            logger.warning("No results to generate report from")
            return {}
        
        # Calculate aggregate statistics
        metrics_list = [r.metrics for r in results]
        
        report = {
            "summary": {
                "total_evaluations": len(results),
                "average_scores": {
                    "retrieval_precision": sum(m.retrieval_precision for m in metrics_list) / len(metrics_list),
                    "retrieval_recall": sum(m.retrieval_recall for m in metrics_list) / len(metrics_list),
                    "answer_relevance": sum(m.answer_relevance for m in metrics_list) / len(metrics_list),
                    "answer_correctness": sum(m.answer_correctness for m in metrics_list) / len(metrics_list),
                    "answer_completeness": sum(m.answer_completeness for m in metrics_list) / len(metrics_list),
                    "citation_accuracy": sum(m.citation_accuracy for m in metrics_list) / len(metrics_list),
                    "citation_coverage": sum(m.citation_coverage for m in metrics_list) / len(metrics_list),
                    "overall_score": sum(m.overall_score for m in metrics_list) / len(metrics_list)
                },
                "score_distribution": {
                    "excellent (>0.9)": sum(1 for m in metrics_list if m.overall_score > 0.9),
                    "good (0.7-0.9)": sum(1 for m in metrics_list if 0.7 <= m.overall_score <= 0.9),
                    "fair (0.5-0.7)": sum(1 for m in metrics_list if 0.5 <= m.overall_score < 0.7),
                    "poor (<0.5)": sum(1 for m in metrics_list if m.overall_score < 0.5)
                }
            },
            "detailed_results": [r.to_dict() for r in results]
        }
        
        # Save to file if requested
        if output_path:
            import json
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Report saved to: {output_path}")
        
        return report