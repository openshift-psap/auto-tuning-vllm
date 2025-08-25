#!/usr/bin/env python3
"""
Basic usage example for auto-tune-vllm.

This example shows how to set up and run a vLLM optimization study
using the Python API instead of the CLI.
"""

import optuna
from auto_tune_vllm import (
    StudyController,
    StudyConfig, 
    RayExecutionBackend
)

def main():
    # Create study configuration
    config = StudyConfig.from_file("study_config.yaml")
    
    # Setup Optuna study with PostgreSQL
    study = optuna.create_study(
        storage=config.database_url,
        study_name="my_vllm_study",
        direction="maximize",  # or ["maximize", "minimize"] for multi-objective
        load_if_exists=True
    )
    
    # Choose execution backend
    # Option 1: Ray distributed execution
    backend = RayExecutionBackend(resource_requirements={"GPU": 1, "CPU": 4})
    
    # Option 2: Local execution for testing
    # backend = LocalExecutionBackend(max_concurrent=2)
    
    # Create study controller
    controller = StudyController(backend=backend, study=study, config=config)
    
    # Run optimization
    print("Starting vLLM optimization study...")
    controller.run_optimization(n_trials=100)
    
    # Print results
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

if __name__ == "__main__":
    main()