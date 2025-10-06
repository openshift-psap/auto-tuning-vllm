#!/usr/bin/env python3
"""Quick test to verify the package can be imported and basic functionality works."""

import sys
from pathlib import Path

# Add the package to Python path for testing
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all major components can be imported."""
    print("Testing imports...")

    # Core/public API
    import auto_tune_vllm  # noqa: F401
    from auto_tune_vllm.core import trial as _trial  # noqa: F401
    print("✓ Core components imported successfully")

    # Configuration system
    from auto_tune_vllm.benchmarks.config import BenchmarkConfig as _BC  # noqa: F401
    print("✓ Configuration system imported successfully")

    # Execution backends
    from auto_tune_vllm.execution.backends import (
        LocalExecutionBackend as _LEB,  # noqa: F401
    )
    print("✓ Execution system imported successfully")

    # Benchmark providers
    from auto_tune_vllm.benchmarks import providers as _providers  # noqa: F401
    print("✓ Benchmark system imported successfully")

    # Logging system
    from auto_tune_vllm.logging import manager as _logging  # noqa: F401
    print("✓ Logging system imported successfully")

    # CLI surface
    import auto_tune_vllm.cli as _cli  # noqa: F401
    print("✓ CLI imported successfully")
def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    from auto_tune_vllm.benchmarks.config import BenchmarkConfig
    
    # Test benchmark config
    benchmark_config = BenchmarkConfig(
        model="test-model",
        max_seconds=60,
        concurrency=10
    )
    assert benchmark_config.use_synthetic_data
    print("✓ BenchmarkConfig works")

def test_backend_interface():
    """Test execution backend interfaces."""
    print("\nTesting execution backend interface...")
    
    from auto_tune_vllm.benchmarks.config import BenchmarkConfig
    from auto_tune_vllm.core.trial import TrialConfig
    from auto_tune_vllm.execution.backends import LocalExecutionBackend
    
    # Create local backend
    backend = LocalExecutionBackend(max_concurrent=1)
    print(
        f"✓ LocalExecutionBackend created with {backend.max_concurrent} "
        f"concurrent slots"
    )
    try:
        # Create trial config
        trial_config = TrialConfig(
            study_id=1,
            trial_number=1,
            parameters={"max_num_batched_tokens": 8192},
            benchmark_config=BenchmarkConfig()
        )
        # Test vLLM args generation
        args = trial_config.vllm_args
        assert "--max-num-batched-tokens" in args
        assert "8192" in args
        print("✓ TrialConfig vLLM args generation works")
    finally:
        backend.shutdown()

def main():
    """Run all tests."""
    print("🚀 Testing auto-tune-vllm package installation\n")
    
    try:
        test_imports()
        test_config_validation()
        test_backend_interface()
        
        print("\n✅ All tests passed! Package is ready for use.")
        print("\nNext steps:")
        print("1. Install the package: pip install -e .")
        print("2. Test CLI: auto-tune-vllm --help")
        print(
            "3. Create a study config and run: auto-tune-vllm optimize "
            "--config study.yaml"
        )
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()