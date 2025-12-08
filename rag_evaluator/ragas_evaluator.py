"""
RAGAS Integration
Integration with RAGAS framework for additional evaluation metrics.
"""

from typing import List, Dict, Optional
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("RAGAS not available. Install with: pip install ragas")
    RAGAS_AVAILABLE = False


class RAGASEvaluator:
    """
    RAGAS framework integration for RAG evaluation.
    Provides additional metrics beyond LLM-as-Judge.
    """
    
    def __init__(self):
        """Initialize RAGAS evaluator."""
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not installed. Install with: pip install ragas")
        
        logger.info("RAGASEvaluator initialized")
    
    def evaluate_with_ragas(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (each question has multiple context chunks)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary with RAGAS evaluation results
        """
        try:
            logger.info(f"Running RAGAS evaluation on {len(questions)} samples")
            
            # Prepare dataset
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            
            if ground_truths:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # Define metrics
            metrics = [
                faithfulness,
                answer_relevancy,
            ]
            
            # Add metrics that require ground truth
            if ground_truths:
                metrics.extend([
                    context_precision,
                    context_recall,
                ])
            
            # Run evaluation
            result = evaluate(dataset, metrics=metrics)
            
            logger.info("RAGAS evaluation complete")
            
            return {
                "ragas_scores": result,
                "summary": self._summarize_ragas_results(result)
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _summarize_ragas_results(self, result) -> Dict:
        """Summarize RAGAS results."""
        try:
            summary = {}
            
            # Extract scores
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for col in df.columns:
                    if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                        summary[col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max())
                        }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to summarize RAGAS results: {e}")
            return {}