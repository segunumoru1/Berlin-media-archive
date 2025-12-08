"""
Test Cases for RAG Evaluation
Provides sample test cases and utilities for creating evaluation datasets.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TestCase:
    """Represents a single evaluation test case."""
    question: str
    answer: str
    context: List[str]
    ground_truth: Optional[str] = None
    expected_grade: Optional[str] = None
    description: str = ""


class EvaluationTestCases:
    """Collection of test cases for RAG evaluation."""
    
    @staticmethod
    def get_sample_test_cases() -> List[Dict]:
        """
        Get sample test cases for demonstration.
        
        Returns:
            List of test case dictionaries
        """
        test_cases = [
            {
                "question": "What is the primary definition of success discussed in the files?",
                "answer": "Success is defined as achieving set goals while maintaining work-life balance. [Source: Document, Page 3] The interview also mentions that success includes personal growth and helping others achieve their potential. [Source: Interview, 04:23]",
                "context": [
                    "[Document, Page 3] Success can be defined in multiple ways, but primarily it involves achieving the goals you set for yourself while maintaining a healthy work-life balance.",
                    "[Interview, 04:23] Host: What does success mean to you? Guest: For me, success isn't just about achievements. It's about personal growth and helping others reach their potential."
                ],
                "ground_truth": "Success is achieving goals with work-life balance and includes personal growth and helping others.",
                "expected_grade": "A",
                "description": "Well-cited answer with specific references"
            },
            {
                "question": "When was the Berlin Wall constructed?",
                "answer": "The Berlin Wall was built in 1961 to divide East and West Berlin. [Source: Document, Page 12]",
                "context": [
                    "[Document, Page 12] Construction of the Berlin Wall began on August 13, 1961, following increasing tensions between East and West Berlin."
                ],
                "ground_truth": "The Berlin Wall was constructed on August 13, 1961.",
                "expected_grade": "A",
                "description": "Factually accurate with citation"
            },
            {
                "question": "What were the main topics discussed in the 1990 interview?",
                "answer": "The interview covered reunification and cultural identity. The speaker mentioned challenges of bringing two societies together. [Source: Interview, 12:45]",
                "context": [
                    "[Interview, 12:45] We discussed the challenges of reunification, particularly how to bring together two societies that had been separated for nearly 30 years.",
                    "[Interview, 18:30] Cultural identity was a major theme, with concerns about preserving Eastern traditions while embracing unity."
                ],
                "ground_truth": "Main topics included reunification challenges and cultural identity preservation.",
                "expected_grade": "B",
                "description": "Good answer but could reference more sources"
            },
            {
                "question": "What is quantum computing?",
                "answer": "Quantum computing uses quantum mechanics principles to process information much faster than classical computers. It leverages superposition and entanglement.",
                "context": [
                    "[Document, Page 5] Traditional computers use bits, which can be 0 or 1. The development of faster processors continues."
                ],
                "ground_truth": "Quantum computing uses quantum mechanics for faster information processing.",
                "expected_grade": "F",
                "description": "Hallucination - context doesn't support quantum computing claims"
            },
            {
                "question": "What did Christa Wolf say about reunification?",
                "answer": "Christa Wolf discussed various topics in her writings and interviews over the years.",
                "context": [
                    "[Interview, 08:15] Christa Wolf: Reunification brings hope, but also anxiety. We must not forget the experiences of those who lived through the division.",
                    "[Document, Page 45] In her 1990 interview, Wolf expressed concerns about rapid reunification and the loss of Eastern identity."
                ],
                "ground_truth": "Wolf expressed both hope and anxiety about reunification, with concerns about rapid change.",
                "expected_grade": "D",
                "description": "Vague answer without using available context"
            }
        ]
        
        return test_cases
    
    @staticmethod
    def create_test_case(
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None,
        expected_grade: Optional[str] = None,
        description: str = ""
    ) -> Dict:
        """
        Create a custom test case.
        
        Args:
            question: The question
            answer: Generated answer
            context: List of context chunks
            ground_truth: Optional ground truth answer
            expected_grade: Optional expected grade
            description: Description of test case
            
        Returns:
            Test case dictionary
        """
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "ground_truth": ground_truth,
            "expected_grade": expected_grade,
            "description": description
        }