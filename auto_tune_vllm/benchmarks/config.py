"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    benchmark_type: str = "guidellm"  # "guidellm" or custom provider name
    model: str = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    max_seconds: int = 300
    dataset: Optional[str] = None  # HF dataset or file path
    prompt_tokens: int = 1000  # For synthetic data
    output_tokens: int = 1000  # For synthetic data
    concurrency: int = 50  # Benchmark concurrency level (legacy, use rates instead)
    
    # Advanced GuideLLM parameters
    processor: Optional[str] = None  # Processor model, defaults to model if not set
    rate: int = 50  # Single rate value for concurrent requests
    samples: int = 1000  # Number of samples to take
    
    # Token statistics for synthetic data - only used when explicitly specified
    prompt_tokens_stdev: Optional[int] = None
    prompt_tokens_min: Optional[int] = None  
    prompt_tokens_max: Optional[int] = None
    output_tokens_stdev: Optional[int] = None
    output_tokens_min: Optional[int] = None
    output_tokens_max: Optional[int] = None
    
    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None