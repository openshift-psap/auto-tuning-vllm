"""Study controller for orchestrating optimization studies."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import optuna
from optuna.samplers import (
    GridSampler, 
    NSGAIISampler,
    RandomSampler,
    TPESampler
)
import optuna.integration

from ..execution.backends import ExecutionBackend, JobHandle
from .config import StudyConfig
from .trial import TrialConfig
from .db_utils import create_database_if_not_exists, verify_database_connection
from ..logging.manager import CentralizedLogger

logger = logging.getLogger(__name__)


class StudyController:
    """Main orchestration controller for optimization studies."""
    
    def __init__(
        self, 
        backend: ExecutionBackend, 
        study: optuna.Study, 
        config: StudyConfig
    ):
        self.backend = backend
        self.study = study
        self.config = config
        self.active_trials: Dict[int, JobHandle] = {}
        self.completed_trials = 0
        
    @staticmethod
    def get_study_id(study_name: str) -> int:
        """Get consistent study ID from study name."""
        # Use a safe hash that fits within PostgreSQL INTEGER range (-2^31 to 2^31-1)
        # Take absolute value and constrain to positive range (0 to 2^31-1)
        return abs(hash(study_name)) % 2147483647
    
    @classmethod
    def create_from_config(
        cls, 
        backend: ExecutionBackend, 
        config: StudyConfig,
        create_db: bool = False
    ) -> StudyController:
        """Create study controller with Optuna study from configuration."""
        # Handle database creation if requested and using PostgreSQL
        if create_db and config.database_url:
            try:
                logger.info("Checking database existence and creating if necessary...")
                db_created = create_database_if_not_exists(config.database_url)
                if db_created:
                    logger.info("Database created successfully")
                else:
                    logger.info("Database already exists")
            except Exception as e:
                logger.error(f"Failed to create database: {e}")
                raise RuntimeError(f"Database creation failed: {e}")
        
        # Verify database connection before proceeding (only if using PostgreSQL)
        if config.database_url and not verify_database_connection(config.database_url):
            if create_db:
                raise RuntimeError(
                    f"Cannot connect to database after creation attempt. "
                    f"Please check your database URL: {config.database_url}"
                )
            else:
                raise RuntimeError(
                    f"Cannot connect to database: {config.database_url}. "
                    f"Database may not exist. Use --create-db flag to create it automatically."
                )
        
        # Create sampler based on config
        sampler = cls._create_sampler(config)
        
        # Determine optimization direction(s)
        if isinstance(config.optimization.objective, list):
            directions = config.optimization.objective
        else:
            directions = config.optimization.objective
        
        # Determine storage backend for Optuna study
        if config.database_url:
            storage = config.database_url
            storage_type = "PostgreSQL"
        elif config.storage_file:
            # Ensure directory exists for file-based storage
            storage_path = Path(config.storage_file)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{config.storage_file}"
            storage_type = "SQLite file"
        else:
            # This should not happen due to validation in config.py, but fallback to in-memory
            storage = None
            storage_type = "in-memory"
        
        logger.info(f"Using {storage_type} storage for Optuna study")
        if storage:
            logger.info(f"Storage location: {storage}")
        
        # Create Optuna study
        try:
            study = optuna.create_study(
                storage=storage,
                study_name=config.study_name,
                direction=directions,  # "maximize", "minimize", or ["maximize", "minimize"] 
                sampler=sampler,
                load_if_exists=True
            )
        except Exception as e:
            if config.database_url and ("does not exist" in str(e).lower() or "database" in str(e).lower()):
                raise RuntimeError(
                    f"Failed to create Optuna study. Database connection error: {e}. "
                    f"Use --create-db flag to create the database automatically."
                )
            else:
                raise RuntimeError(f"Failed to create Optuna study: {e}")
        
        # Generate and prominently log study ID
        study_id = cls.get_study_id(config.study_name)
        logger.info(f"Created study: {config.study_name} with {config.optimization.sampler} sampler")
        logger.info(f"🔍 STUDY ID: {study_id} (use this ID for log viewing)")
        
        # Initialize logging infrastructure if configured
        log_database_url = None
        log_file_path = None
        
        if config.logging_config:
            log_database_url = config.logging_config.get("database_url")
            log_file_path = config.logging_config.get("file_path")
            
        # Default to main database if no specific logging config and database is available
        if not log_database_url and not log_file_path and config.database_url:
            log_database_url = config.database_url
        elif not log_database_url and not log_file_path and not config.database_url:
            # No PostgreSQL available - enforce file logging mode
            log_file_path = f"./logs/study_{study_id}"
            logger.info(f"No PostgreSQL available - using file logging: {log_file_path}")
        
        try:
            # Initialize CentralizedLogger
            CentralizedLogger(
                study_id=study_id,
                pg_url=log_database_url,
                file_path=log_file_path,
                log_level=config.logging_config.get("log_level", "INFO") if config.logging_config else "INFO"
            )
            
            # Provide appropriate log viewing instructions
            if log_file_path:
                logger.info(f"📋 File logging enabled. Logs will be written to: {log_file_path}")
                logger.info(f"📋 View logs with: auto-tune-vllm view-file-logs --study-id {study_id} --log-path {log_file_path}")
            elif log_database_url:
                logger.info(f"📋 Database logging ready. View logs with: auto-tune-vllm logs --study-id {study_id} --database-url {log_database_url}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize logging infrastructure: {e}")
            if log_file_path:
                logger.info(f"📋 To view file logs: auto-tune-vllm view-file-logs --study-id {study_id} --log-path {log_file_path}")
            elif log_database_url:
                logger.info(f"📋 To view logs: auto-tune-vllm logs --study-id {study_id} --database-url {log_database_url}")
            else:
                logger.info("📋 Console logging only - no file or database logging configured")
        
        return cls(backend, study, config)
    
    @staticmethod
    def _create_sampler(config: StudyConfig) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        sampler_name = config.optimization.sampler.lower()
        
        if sampler_name == "tpe":
            return TPESampler()
        elif sampler_name == "random":
            return RandomSampler()
        elif sampler_name == "botorch":
            return optuna.integration.BoTorchSampler()
        elif sampler_name == "nsga2":
            return NSGAIISampler()
        elif sampler_name == "grid":
            # Build search space for grid sampler
            search_space = {}
            for param_name, param_config in config.parameters.items():
                if param_config.enabled:
                    # Convert parameter config to grid search space
                    if hasattr(param_config, 'options'):
                        search_space[param_name] = param_config.options
                    elif hasattr(param_config, 'min_value'):
                        # Generate discrete values for range parameters
                        values = []
                        current = param_config.min_value
                        while current <= param_config.max_value:
                            values.append(current)
                            current += param_config.step or 1
                        search_space[param_name] = values
                    else:
                        search_space[param_name] = [True, False]  # Boolean
            
            grid_size = StudyController._calculate_grid_size(search_space)
            logger.info(f"Grid search space: {len(search_space)} parameters, "
                       f"{grid_size} total combinations")
            return GridSampler(search_space)
        else:
            logger.warning(f"Unknown sampler: {sampler_name}, using TPE")
            return TPESampler()
    
    @staticmethod
    def _calculate_grid_size(search_space: Dict) -> int:
        """Calculate total grid search combinations."""
        size = 1
        for values in search_space.values():
            size *= len(values)
        return size
    
    def run_optimization(
        self, 
        n_trials: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        poll_interval: float = 5.0
    ) -> optuna.Study:
        """
        Run optimization study.
        
        Args:
            n_trials: Number of trials to run (overrides config)
            max_concurrent: Maximum concurrent trials (None = unlimited)
            poll_interval: How often to poll for completed trials (seconds)
            
        Returns:
            Completed Optuna study
        """
        total_trials = n_trials or self.config.optimization.n_trials
        max_concurrent = max_concurrent or float('inf')
        
        logger.info(f"Starting optimization: {total_trials} trials, "
                   f"max concurrent: {max_concurrent if max_concurrent != float('inf') else 'unlimited'}")
        
        try:
            while self.completed_trials < total_trials:
                # Submit new trials up to concurrency limit
                self._submit_available_trials(
                    remaining_trials=total_trials - self.completed_trials - len(self.active_trials),
                    max_concurrent=max_concurrent
                )
                
                # Poll for completed trials
                if self.active_trials:
                    newly_completed = self._collect_completed_trials()
                    self.completed_trials += newly_completed
                    
                    if newly_completed > 0:
                        logger.info(f"Progress: {self.completed_trials}/{total_trials} trials completed")
                
                # Sleep before next poll cycle
                time.sleep(poll_interval)
            
            # Wait for any remaining trials
            while self.active_trials:
                newly_completed = self._collect_completed_trials()
                self.completed_trials += newly_completed
                
                if newly_completed > 0:
                    logger.info(f"Final cleanup: {self.completed_trials} trials completed")
                
                if not self.active_trials:
                    break
                    
                time.sleep(poll_interval)
            
            logger.info(f"Optimization completed: {self.completed_trials} trials")
            return self.study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        finally:
            self.backend.shutdown()
    
    def _submit_available_trials(self, remaining_trials: int, max_concurrent: float):
        """Submit new trials up to limits."""
        while (remaining_trials > 0 and 
               len(self.active_trials) < max_concurrent):
            
            # Ask Optuna for next trial
            try:
                trial = self.study.ask()
            except Exception as e:
                logger.error(f"Failed to get next trial from Optuna: {e}")
                break
            
            # Build trial configuration
            trial_config = self._build_trial_config(trial)
            
            # Submit to execution backend
            try:
                job_handle = self.backend.submit_trial(trial_config)
                self.active_trials[trial.number] = job_handle
                
                logger.info(f"Submitted trial {trial.number} with parameters: {trial.params}")
                remaining_trials -= 1
                
            except Exception as e:
                logger.error(f"Failed to submit trial {trial.number}: {e}")
                # Mark trial as failed in Optuna
                self.study.tell(trial.number, None)  # Failed trial
                break
    
    def _collect_completed_trials(self) -> int:
        """Collect completed trials and report results to Optuna."""
        if not self.active_trials:
            return 0
        
        # Poll for completed trials
        job_handles = list(self.active_trials.values())
        completed_results, remaining_handles = self.backend.poll_trials(job_handles)
        
        completed_count = 0
        
        for result in completed_results:
            trial_number = result.trial_number
            
            # Remove from active trials
            if trial_number in self.active_trials:
                del self.active_trials[trial_number]
            
            # Report to Optuna
            try:
                if result.success and result.objective_values:
                    self.study.tell(trial_number, result.objective_values)
                    logger.info(f"Trial {trial_number} completed successfully: {result.objective_values}")
                else:
                    # Failed trial
                    self.study.tell(trial_number, None)
                    logger.error(f"Trial {trial_number} failed: {result.error_message}")
                
                completed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to report trial {trial_number} to Optuna: {e}")
        
        return completed_count
    
    def _build_trial_config(self, trial: optuna.Trial) -> TrialConfig:
        """Build trial configuration from Optuna trial."""
        # Generate parameter values using configured parameter definitions
        parameters = {}
        
        for param_name, param_config in self.config.parameters.items():
            if param_config.enabled:
                value = param_config.generate_optuna_suggest(trial)
                parameters[param_name] = value
        
        return TrialConfig(
            study_id=self.get_study_id(self.config.study_name),
            trial_number=trial.number,
            parameters=parameters,
            benchmark_config=self.config.benchmark,
            optimization_config=self.config.optimization,
            logging_config=self.config.logging_config
        )
    
    def get_optimization_results(self) -> Dict:
        """Get optimization results summary."""
        if self.config.optimization.is_multi_objective:
            # Multi-objective results
            return {
                "type": "multi_objective",
                "approach": self.config.optimization.approach,
                "objectives": [
                    {"metric": obj.metric, "direction": obj.direction, "percentile": obj.percentile}
                    for obj in self.config.optimization.objectives
                ],
                "n_trials": len(self.study.trials),
                "n_pareto_solutions": len(self.study.best_trials),
                "pareto_front": [
                    {"trial": t.number, "values": t.values, "params": t.params} 
                    for t in self.study.best_trials[:10]  # Top 10
                ]
            }
        else:
            # Single objective results
            best_trial = self.study.best_trial
            objective = self.config.optimization.objectives[0]
            return {
                "type": "single_objective",
                "approach": self.config.optimization.approach,
                "objective": {
                    "metric": objective.metric,
                    "direction": objective.direction,
                    "percentile": objective.percentile
                },
                "n_trials": len(self.study.trials),
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_trial_number": best_trial.number
            }
    
    def resume_study(self) -> StudyController:
        """Resume an existing study from the database."""
        study_id = self.get_study_id(self.config.study_name)
        logger.info(f"Resuming study: {self.config.study_name}")
        logger.info(f"🔍 STUDY ID: {study_id} (use this ID for log viewing)")
        
        # Provide appropriate log viewing instructions based on config
        if self.config.logging_config and self.config.logging_config.get("file_path"):
            log_file_path = self.config.logging_config["file_path"]
            logger.info(f"📋 View file logs with: auto-tune-vllm view-file-logs --study-id {study_id} --log-path {log_file_path}")
        elif self.config.database_url:
            logger.info(f"📋 View logs with: auto-tune-vllm logs --study-id {study_id} --database-url {self.config.database_url}")
        else:
            logger.info(f"📋 Using file-based storage. View logs with: auto-tune-vllm view-file-logs --study-id {study_id} --log-path ./logs/study_{study_id}")
        
        # Count existing trials
        n_existing = len(self.study.trials)
        logger.info(f"Found {n_existing} existing trials in study")
        
        return self