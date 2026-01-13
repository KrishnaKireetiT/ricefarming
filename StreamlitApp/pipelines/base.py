"""
Abstract base class for pipeline implementations.
All pipelines must implement this interface to be swappable in the app.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Structured result from pipeline execution."""
    question: str
    farmer_answer: str
    raw_entities: List[str]
    aligned_entities: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]
    vector_context: List[Dict[str, Any]]
    keyword_results: List[Dict[str, Any]]

    trace_id: str
    trace_url: str
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "farmer_answer": self.farmer_answer,
            "raw_entities": self.raw_entities,
            "aligned_entities": self.aligned_entities,
            "graph_facts": self.graph_facts,
            "vector_context": self.vector_context,
            "keyword_results": self.keyword_results,

            "trace_id": self.trace_id,
            "trace_url": self.trace_url,
            "execution_time": self.execution_time,
        }


class BasePipeline(ABC):
    """
    Abstract base class for all pipeline implementations.
    
    To add a new pipeline:
    1. Create a new file in pipelines/ directory
    2. Extend this class
    3. Implement all abstract methods
    4. Register in pipelines/__init__.py
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this pipeline."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return the version string of this pipeline."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a brief description of this pipeline."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the pipeline (load models, connect to databases).
        Called once when the pipeline is first selected.
        """
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        pass
    
    @abstractmethod
    def run_query(self, question: str) -> PipelineResult:
        """
        Run a single query through the pipeline.
        
        Args:
            question: The user's question
            
        Returns:
            PipelineResult with all pipeline outputs
        """
        pass
    
    def run_batch(self, questions: List[str], progress_callback=None) -> List[PipelineResult]:
        """
        Run multiple queries through the pipeline.
        
        Args:
            questions: List of questions to process
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        for i, question in enumerate(questions):
            result = self.run_query(question)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(questions))
        return results
