"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class PrefixBucketConfig:
    prefix_tokens: int = 0
    prefix_count: int = 1
    bucket_weight: int = 100


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

    # Agentic / multi-turn workload parameters (requires GuideLLM >= 0.6.0).
    # Only relevant for agentic benchmarks; leave unset for regular single-turn
    # GuideLLM behavior.
    # turns: number of sequential turns per synthetic conversation. Each turn
    #   N+1 sends the prior turns' user messages AND actual assistant responses
    #   as conversation history, producing response-conditioned growing context.
    # prefix_buckets: shared synthetic system-prompt prefix(es) prepended to
    #   every request from a bucket — drives prefix-cache hits across
    #   conversations. Use prefix_count=1 for a single shared prefix; >1 for a
    #   pool sampled per conversation. Combine multiple buckets to mix
    #   shared/diverse traffic. Schema validation deferred to GuideLLM.
    turns: Optional[int] = None
    prefix_buckets: Optional[list[PrefixBucketConfig]] = None

    # Set in benchmark section of study config
    # Logging level for GuideLLM
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def __post_init__(self):
        # Coerce dicts (e.g. from YAML) into PrefixBucketConfig.
        if self.prefix_buckets:
            self.prefix_buckets = [
                b if isinstance(b, PrefixBucketConfig) else PrefixBucketConfig(**b)
                for b in self.prefix_buckets
            ]
    
    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None