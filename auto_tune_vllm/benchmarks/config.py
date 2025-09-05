"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    benchmark_type: str = "guidellm"  # "guidellm" or custom provider name
    model: str = "Qwen/Qwen3-30B-A3B-FP8"
    max_seconds: int = 300
    dataset: Optional[str] = None  # HF dataset or file path
    prompt_tokens: int = 1000  # For synthetic data
    output_tokens: int = 1000  # For synthetic data
    concurrency: int = 50  # Benchmark concurrency level (legacy, use rates instead)
    
    # Advanced GuideLLM parameters
    processor: Optional[str] = None  # Processor model, defaults to model if not set
    rate: int = 50  # Single rate value for concurrent requests
    
    # Token statistics for synthetic data
    prompt_tokens_stdev: int = 128
    prompt_tokens_min: int = 256  
    prompt_tokens_max: int = 1024
    output_tokens_stdev: int = 512
    output_tokens_min: int = 1024
    output_tokens_max: int = 3072
    
    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None