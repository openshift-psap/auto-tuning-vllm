"""Study controller for orchestrating optimization studies."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import optuna.integration
from optuna.samplers import (
    GPSampler,
    GridSampler,
    NSGAIISampler,
    RandomSampler,
    TPESampler,
)

from ..execution.backends import ExecutionBackend, JobHandle
from ..logging.manager import CentralizedLogger
from .config import StudyConfig
from .db_utils import create_database_if_not_exists, verify_database_connection
from .parameters import EnvironmentParameter, ListParameter, RangeParameter
from .trial import TrialConfig

logger = logging.getLogger(__name__)

TWO_HOURS_IN_SECONDS = 7200
POLL_RATE = 5


class StudyController:
    """Main orchestration controller for optimization studies."""

    def __init__(
        self, backend: ExecutionBackend, study: optuna.Study, config: StudyConfig
    ):
        self.backend: ExecutionBackend = backend
        self.study: optuna.Study = study
        self.config: StudyConfig = config
        self.active_trials: Dict[str, JobHandle] = {}
        self.completed_trials: int = 0
        self.baseline_results: Dict[
            int, List[float]
        ] = {}  # concurrency -> objective_values

    @staticmethod
    def get_study_identifier(study_name: str) -> str:
        """Get study identifier - now just returns the study name for consistency."""
        return study_name

    @classmethod
    def create_from_config(
        cls, backend: ExecutionBackend, config: StudyConfig, create_db: bool = False
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
                    "Cannot connect to database after creation attempt. "
                    f"Please check your database URL: {config.database_url}"
                )
            else:
                raise RuntimeError(
                    f"Cannot connect to database: {config.database_url}. "
                    "Database may not exist. "
                    "Use --create-db flag to create it automatically."
                )

        # Create sampler based on config
        sampler = cls._create_sampler(config)

        # Determine optimization directions for Optuna
        # (works for single and multi-objective)
        directions = config.optimization.optuna_directions

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
            # This should not happen due to validation in config.py,
            # but fallback to in-memory
            storage = None
            storage_type = "in-memory"

        logger.info(f"Using {storage_type} storage for Optuna study")
        if storage:
            logger.info(f"Storage location: {storage}")

        # Create Optuna study with appropriate load_if_exists behavior
        load_if_exists = (
            not config.use_explicit_name
        )  # For explicit names, fail if exists

        if config.use_explicit_name:
            logger.info(
                "Creating new study with explicit name: "
                f"{config.study_name} (will fail if exists)"
            )
        else:
            logger.info(
                f"Creating study: {config.study_name} (will load existing if found)"
            )

        try:
            study = optuna.create_study(
                storage=storage,
                study_name=config.study_name,
                directions=directions,
                sampler=sampler,
                load_if_exists=load_if_exists,
            )
        except Exception as e:
            # Handle explicit name conflicts with helpful error messages
            if config.use_explicit_name and "already exists" in str(e).lower():
                raise RuntimeError(
                    f"Study '{config.study_name}' already exists. "
                    f"Options: \n"
                    f"  • Use 'auto-tune-vllm resume --config <config_file>' "
                    f"to continue the existing study\n"
                    f"  • Change the study name to create a new study\n"
                    f"  • Use 'prefix: {config.study_name}' instead of 'name' "
                    f"for auto-generated unique names"
                )
            elif config.database_url and (
                "does not exist" in str(e).lower() or "database" in str(e).lower()
            ):
                raise RuntimeError(
                    f"Failed to create Optuna study. Database connection error: {e}. "
                    f"Use --create-db flag to create the database automatically."
                )
            else:
                raise RuntimeError(f"Failed to create Optuna study: {e}")

        # Log study information
        logger.info(
            f"Created study: {config.study_name} "
            f"with {config.optimization.sampler} sampler"
        )
        logger.info(
            f"🔍 Study Name: {config.study_name} (use this name for log viewing)"
        )

        # Initialize logging infrastructure if configured
        log_database_url = None
        log_file_path = None

        if config.logging_config:
            log_database_url = config.logging_config.get("database_url")
            log_file_path = config.logging_config.get("file_path")

        # Default to main database if no specific logging config
        # and database is available
        if not log_database_url and not log_file_path and config.database_url:
            log_database_url = config.database_url
        elif not log_database_url and not log_file_path and not config.database_url:
            # No PostgreSQL available - enforce file logging mode
            log_file_path = f"./logs/{config.study_name}"
            logger.info(
                f"No PostgreSQL available - using file logging: {log_file_path}"
            )

        try:
            # Initialize CentralizedLogger
            CentralizedLogger(
                study_name=config.study_name,
                pg_url=log_database_url,
                file_path=log_file_path,
                log_level=config.logging_config.get("log_level", "INFO")
                if config.logging_config
                else "INFO",
            )

            # Provide appropriate log viewing instructions
            if log_file_path:
                logger.info(
                    f"📋 File logging enabled. Logs will be written to: {log_file_path}"
                )
                logger.info(
                    f"📋 View logs with: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--log-path {log_file_path}"
                )
            elif log_database_url:
                logger.info(
                    f"📋 Database logging ready. View logs with: "
                    f"auto-tune-vllm logs --study-name {config.study_name} "
                    f"--database-url {log_database_url}"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize logging infrastructure: {e}")
            if log_file_path:
                logger.info(
                    f"📋 To view file logs: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--log-path {log_file_path}"
                )
            elif log_database_url:
                logger.info(
                    f"📋 To view logs: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--database-url {log_database_url}"
                )
            else:
                logger.info(
                    "📋 Console logging only - no file or database logging configured"
                )

        return cls(backend, study, config)

    @classmethod
    def resume_from_config(
        cls, backend: ExecutionBackend, config: StudyConfig
    ) -> StudyController:
        """Resume an existing study from configuration. Fails if study doesn't exist."""
        # Verify database connection before proceeding (only if using PostgreSQL)
        if config.database_url and not verify_database_connection(config.database_url):
            raise RuntimeError(
                f"Cannot connect to database: {config.database_url}. "
                f"Please check your database connection."
            )

        # Create sampler based on config
        sampler = cls._create_sampler(config)

        # Determine storage backend for Optuna study
        if config.database_url:
            storage = config.database_url
            storage_type = "PostgreSQL"
        elif config.storage_file:
            # Ensure directory exists for file-based storage
            storage_path = Path(config.storage_file)
            if not storage_path.exists():
                raise RuntimeError(
                    f"Study '{config.study_name}' not found in storage. "
                    f"Storage file does not exist: {config.storage_file}. "
                    f"Options:\n"
                    f"  • Use 'auto-tune-vllm optimize --config <config_file>' "
                    f"to create a new study\n"
                    f"  • Verify the study name and storage path are correct"
                )
            storage = f"sqlite:///{config.storage_file}"
            storage_type = "SQLite file"
        else:
            # This should not happen due to validation in config.py
            raise RuntimeError("No storage configuration found. Cannot resume study.")

        logger.info(f"Using {storage_type} storage for resuming Optuna study")
        logger.info(f"Storage location: {storage}")

        # Try to load existing study - this will fail if the study doesn't exist
        logger.info(f"Attempting to resume existing study: {config.study_name}")

        try:
            study = optuna.load_study(
                storage=storage, study_name=config.study_name, sampler=sampler
            )
        except Exception as e:
            # Provide helpful error messages for common resume failures
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise RuntimeError(
                    f"Study '{config.study_name}' not found in storage. "
                    f"Options:\n"
                    f"  • Use 'auto-tune-vllm optimize --config <config_file>' "
                    f"to create a new study\n"
                    f"  • Verify the study name is correct\n"
                    f"  • Check that the storage location contains the study"
                )
            elif config.database_url and (
                "database" in str(e).lower() or "connection" in str(e).lower()
            ):
                raise RuntimeError(
                    f"Failed to connect to database for resuming study: {e}. "
                    f"Please check your database connection."
                )
            else:
                raise RuntimeError(f"Failed to resume study '{config.study_name}': {e}")

        # Log study information
        logger.info(f"Successfully resumed study: {config.study_name}")
        logger.info(
            f"🔍 Study Name: {config.study_name} (use this name for log viewing)"
        )

        # Initialize logging infrastructure if configured (same as create_from_config)
        log_database_url = None
        log_file_path = None

        if config.logging_config:
            log_database_url = config.logging_config.get("database_url")
            log_file_path = config.logging_config.get("file_path")

        # Default to main database if no specific logging config
        # and database is available
        if not log_database_url and not log_file_path and config.database_url:
            log_database_url = config.database_url
        elif not log_database_url and not log_file_path and not config.database_url:
            # No PostgreSQL available - enforce file logging mode
            log_file_path = f"./logs/{config.study_name}"
            logger.info(
                f"No PostgreSQL available - using file logging: {log_file_path}"
            )

        try:
            # Initialize CentralizedLogger
            CentralizedLogger(
                study_name=config.study_name,
                pg_url=log_database_url,
                file_path=log_file_path,
                log_level=config.logging_config.get("log_level", "INFO")
                if config.logging_config
                else "INFO",
            )

            # Provide appropriate log viewing instructions
            if log_file_path:
                logger.info(
                    f"📋 File logging enabled. Logs will be written to: {log_file_path}"
                )
                logger.info(
                    f"📋 View logs with: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--log-path {log_file_path}"
                )
            elif log_database_url:
                logger.info(
                    f"📋 Database logging ready. View logs with: "
                    f"auto-tune-vllm logs --study-name {config.study_name} "
                    f"--database-url {log_database_url}"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize logging infrastructure: {e}")
            if log_file_path:
                logger.info(
                    f"📋 To view file logs: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--log-path {log_file_path}"
                )
            elif log_database_url:
                logger.info(
                    f"📋 To view logs: auto-tune-vllm logs "
                    f"--study-name {config.study_name} "
                    f"--database-url {log_database_url}"
                )
            else:
                logger.info(
                    "📋 Console logging only - no file or database logging configured"
                )

        return cls(backend, study, config)

    @staticmethod
    def _create_search_space(config: StudyConfig) -> dict:
        search_space = {}
        for param_name, param_config in config.parameters.items():
            if param_config.enabled:
                # Convert parameter config to grid search space
                if isinstance(param_config, ListParameter) or isinstance(
                    param_config, EnvironmentParameter
                ):
                    search_space[param_name] = param_config.options
                elif isinstance(param_config, RangeParameter):
                    values = []
                    min_val = param_config.min_value
                    max_val = param_config.max_value
                    step = param_config.step or 1

                    # Use integer-based generation for floats
                    # to avoid accumulation drift
                    if param_config.data_type == "float":
                        # Calculate number of steps and generate via multiplication
                        n_steps = int((max_val - min_val) / step) + 1
                        # Cap to reasonable grid size
                        n_steps = min(n_steps, 10000)
                        for i in range(n_steps):
                            val = min_val + (i * step)
                            if val > max_val:
                                break
                            # Round to avoid floating precision artifacts
                            values.append(round(val, 3))
                    else:
                        # Integer range - simple iteration
                        current = min_val
                        while current <= max_val:
                            values.append(current)
                            current += step
                            # Safety break for large ranges
                            if len(values) > 10000:
                                break
                    search_space[param_name] = values
                else:
                    search_space[param_name] = [True, False]  # Boolean

        return search_space

    @staticmethod
    def _create_sampler(config: StudyConfig) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        sampler_name = config.optimization.sampler.lower()

        if sampler_name == "tpe":
            return TPESampler()
        elif sampler_name == "random":
            return RandomSampler()
        elif sampler_name == "gp":
            return GPSampler()
        elif sampler_name == "botorch":
            return optuna.integration.BoTorchSampler()
        elif sampler_name == "nsga2":
            return NSGAIISampler()
        elif sampler_name == "grid":
            # Build search space for grid sampler
            search_space = StudyController._create_search_space(config)
            grid_size = StudyController._calculate_grid_size(search_space)
            logger.info(
                f"Grid search space: {len(search_space)} parameters, "
                f"{grid_size} total combinations"
            )
            return GridSampler(search_space)
        else:
            raise NotImplementedError(
                f"Unknown sampler '{sampler_name}'. Supported samplers: "
                "tpe, random, gp, botorch, nsga2, grid"
            )
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
        poll_interval: float = 5.0,
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

        # Require explicit, positive concurrency specification
        if max_concurrent is None:
            msg = (
                "❌ --max-concurrent is required to prevent GPU memory conflicts!\n\n"
                "Add to your YAML config:\n"
                "  optimization:\n"
                "    max_concurrent: 2  # Match your GPU count\n\n"
                "Or use CLI: --max-concurrent 2"
            )
            raise ValueError(msg)
        if max_concurrent < 1:
            raise ValueError("--max-concurrent must be >= 1")

        max_concurrent_str = (
            max_concurrent if max_concurrent != float("inf") else "unlimited"
        )
        logger.info(
            f"Starting optimization: {total_trials} trials, "
            f"max concurrent: {max_concurrent_str}"
        )

        try:
            # Run baseline trials first if configured
            if self.config.baseline and self.config.baseline.enabled:
                self._run_baseline_trials()

            while self.completed_trials < total_trials:
                # Submit new trials up to concurrency limit
                self._submit_available_trials(
                    remaining_trials=total_trials
                    - self.completed_trials
                    - len(self.active_trials),
                    max_concurrent=max_concurrent,
                )

                # Poll for completed trials
                if self.active_trials:
                    newly_completed = self._collect_completed_trials()
                    self.completed_trials += newly_completed

                    if newly_completed > 0:
                        logger.info(
                            f"Progress: {self.completed_trials}/{total_trials} "
                            f"trials completed"
                        )

                # Sleep before next poll cycle
                time.sleep(poll_interval)

            # Wait for any remaining trials
            final_cleanup_attempts = 0
            max_cleanup_attempts = (
                60  # Prevent infinite loop (5 minutes at 5s intervals)
            )

            while self.active_trials and final_cleanup_attempts < max_cleanup_attempts:
                newly_completed = self._collect_completed_trials()
                self.completed_trials += newly_completed

                if newly_completed > 0:
                    logger.info(
                        f"Final cleanup: {self.completed_trials} trials completed"
                    )

                if not self.active_trials:
                    break

                final_cleanup_attempts += 1
                if final_cleanup_attempts % 12 == 0:  # Every minute
                    logger.warning(
                        f"Still waiting for {len(self.active_trials)} "
                        f"active trials to complete "
                        f"(attempt {final_cleanup_attempts}/"
                        f"{max_cleanup_attempts})"
                    )
                    logger.debug(f"Active trial IDs: {list(self.active_trials.keys())}")

                time.sleep(poll_interval)

            # If we hit the max attempts, log an error but continue
            if final_cleanup_attempts >= max_cleanup_attempts and self.active_trials:
                logger.error(
                    f"Timeout waiting for {len(self.active_trials)} "
                    f"active trials to complete. "
                    f"Forcing completion. "
                    f"Active trial IDs: {list(self.active_trials.keys())}"
                )
                self.active_trials.clear()  # Force clear to prevent infinite loop

            logger.info(f"Optimization completed: {self.completed_trials} trials")
            return self.study

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        finally:
            # Clean up all active trials before shutting down the backend
            try:
                logger.info("Initiating cleanup of all active trials...")
                self.backend.cleanup_all_trials()
            except Exception as cleanup_e:
                logger.error(f"Error during trial cleanup: {cleanup_e}")
            finally:
                self.backend.shutdown()

    def _submit_available_trials(self, remaining_trials: int, max_concurrent: float):
        """Submit new trials up to limits."""
        while remaining_trials > 0 and len(self.active_trials) < max_concurrent:
            # Ask Optuna for next trial
            try:
                trial = self.study.ask()
            except Exception as e:
                logger.error(f"Failed to get next trial from Optuna: {e}")
                break

            trial_config = self._build_trial_config(trial)

            try:
                job_handle = self.backend.submit_trial(trial_config)
                self.active_trials[trial_config.trial_id] = job_handle

                logger.info(
                    f"Submitted trial {trial.number} with parameters: {trial.params}"
                )
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
        completed_results, _ = self.backend.poll_trials(job_handles)

        optimization_completed_count = 0

        for result in completed_results:
            trial_id = result.trial_id

            # Remove from active trials
            if trial_id in self.active_trials:
                del self.active_trials[trial_id]

            # Report to Optuna (only for optimization trials)
            try:
                if (
                    result.trial_type == "optimization"
                    and result.trial_number is not None
                ):
                    if result.success and result.objective_values:
                        self.study.tell(result.trial_number, result.objective_values)
                        logger.info(
                            f"Trial {trial_id} completed successfully: "
                            f"{result.objective_values}"
                        )
                    else:
                        # Failed trial
                        self.study.tell(result.trial_number, None)
                        logger.error(f"Trial {trial_id} failed: {result.error_message}")

                    # Only count optimization trials toward completion
                    optimization_completed_count += 1

                elif result.trial_type == "baseline":
                    baseline_result = (
                        result.objective_values if result.success else "failed"
                    )
                    logger.info(
                        f"Baseline trial {trial_id} completed: {baseline_result}"
                    )
                    # Baseline trials don't count toward optimization progress

            except Exception as e:
                logger.error(f"Failed to report trial {trial_id} to Optuna: {e}")

        return optimization_completed_count

    def _build_trial_config(self, trial: optuna.Trial) -> TrialConfig:
        """Build trial configuration from Optuna trial."""
        # Start with static parameters that apply to all trials
        parameters = self.config.static_parameters.copy()

        # Add optimizable parameter values using configured parameter definitions
        for param_name, param_config in self.config.parameters.items():
            if param_config.enabled:
                value = param_config.generate_optuna_suggest(trial)
                parameters[param_name] = value

        return TrialConfig(
            study_name=self.config.study_name,
            trial_id=f"trial_{trial.number}",
            trial_number=trial.number,
            trial_type="optimization",
            parameters=parameters,
            parameter_configs=self.config.parameters,
            static_environment_variables=self.config.static_environment_variables,
            benchmark_config=self.config.benchmark,
            optimization_config=self.config.optimization,
            logging_config=self.config.logging_config,
        )

    def get_best_baseline_result(self) -> Optional[List[float]]:
        """Get the best baseline result for comparison."""
        if not self.baseline_results:
            return None

        # For single objective, find the best baseline based on optimization direction
        if len(self.config.optimization.objectives) == 1:
            objective = self.config.optimization.objectives[0]
            if objective.direction == "maximize":
                best_concurrency = max(
                    self.baseline_results.keys(),
                    key=lambda k: self.baseline_results[k][0],
                )
            else:  # minimize
                best_concurrency = min(
                    self.baseline_results.keys(),
                    key=lambda k: self.baseline_results[k][0],
                )
            return self.baseline_results[best_concurrency]
        else:
            # For multi-objective, return the first baseline
            # could be improved with Pareto analysis
            first_concurrency = min(self.baseline_results.keys())
            return self.baseline_results[first_concurrency]

    def get_optimization_results(self) -> Dict:
        """Get optimization results summary."""
        # Get baseline results for comparison
        baseline_result = self.get_best_baseline_result()

        if self.config.optimization.is_multi_objective:
            # Multi-objective results
            pareto_front = []
            for t in self.study.best_trials[:10]:  # Top 10
                trial_data = {"trial": t.number, "values": t.values, "params": t.params}

                # Add baseline comparison for each objective
                if baseline_result:
                    improvements = []
                    for i, (value, objective) in enumerate(
                        zip(t.values, self.config.optimization.objectives)
                    ):
                        baseline_val = (
                            baseline_result[i] if i < len(baseline_result) else None
                        )
                        if baseline_val and baseline_val != 0:
                            if objective.direction == "maximize":
                                improvement = (
                                    (value - baseline_val) / baseline_val
                                ) * 100
                            else:  # minimize
                                improvement = (
                                    (baseline_val - value) / baseline_val
                                ) * 100
                            improvements.append(improvement)
                        else:
                            improvements.append(None)
                    trial_data["baseline_improvements"] = improvements

                pareto_front.append(trial_data)

            return {
                "type": "multi_objective",
                "approach": self.config.optimization.approach,
                "objectives": [
                    {
                        "metric": obj.metric,
                        "direction": obj.direction,
                        "percentile": obj.percentile,
                    }
                    for obj in self.config.optimization.objectives
                ],
                "n_trials": len(self.study.trials),
                "n_pareto_solutions": len(self.study.best_trials),
                "pareto_front": pareto_front,
                "baseline_values": baseline_result,
            }
        else:
            # Single objective results
            best_trial = self.study.best_trial
            objective = self.config.optimization.objectives[0]

            # Calculate baseline improvement
            baseline_improvement = None
            if baseline_result and len(baseline_result) > 0 and baseline_result[0] != 0:
                baseline_val = baseline_result[0]
                if objective.direction == "maximize":
                    baseline_improvement = (
                        (best_trial.value - baseline_val) / baseline_val
                    ) * 100
                else:  # minimize
                    baseline_improvement = (
                        (baseline_val - best_trial.value) / baseline_val
                    ) * 100

            return {
                "type": "single_objective",
                "approach": self.config.optimization.approach,
                "objective": {
                    "metric": objective.metric,
                    "direction": objective.direction,
                    "percentile": objective.percentile,
                },
                "n_trials": len(self.study.trials),
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_trial_number": best_trial.number,
                "baseline_value": baseline_result[0] if baseline_result else None,
                "baseline_improvement": baseline_improvement,
            }

    def resume_study(self) -> StudyController:
        """Resume an existing study from the database."""
        logger.info(f"Resuming study: {self.config.study_name}")
        logger.info(
            f"🔍 Study Name: {self.config.study_name} (use this name for log viewing)"
        )
        log_database_url = None
        log_file_path = None

        if self.config.logging_config:
            log_database_url = self.config.logging_config.get("database_url")
            log_file_path = self.config.logging_config.get("file_path")

        if not log_database_url and not log_file_path and self.config.database_url:
            log_database_url = self.config.database_url
        elif (
            not log_database_url and not log_file_path and not self.config.database_url
        ):
            # No PostgreSQL available - use file logging
            log_file_path = f"./logs/{self.config.study_name}"

        # Display appropriate instructions using the unified logs command
        if log_file_path:
            logger.info(
                f"📋 View logs with: auto-tune-vllm logs "
                f"--study-name {self.config.study_name} "
                f"--log-path {log_file_path}"
            )
        elif log_database_url:
            logger.info(
                f"📋 View logs with: auto-tune-vllm logs "
                f"--study-name {self.config.study_name} "
                f"--database-url {log_database_url}"
            )
        else:
            logger.info(
                "📋 Console logging only - no database or file logging configured"
            )

        # Count existing trials
        n_existing = len(self.study.trials)
        logger.info(f"Found {n_existing} existing trials in study")

        return self

    def _run_baseline_trials(self):
        """Run baseline trials using pure vLLM defaults.

        Adds max-num-seqs when concurrency > 256.
        """
        logger.info("🔄 Running baseline trials...")

        for concurrency in self.config.baseline.concurrency_levels:
            logger.info(f"Running baseline trial with concurrency={concurrency}")

            # Create baseline trial config with custom parameters from config and adds
            # max-num-seqs when needed. Start with static parameters that apply to all
            # trials
            baseline_parameters = self.config.static_parameters.copy()
            # Add baseline-specific parameters from config (if any)
            if self.config.baseline.parameters:
                baseline_parameters.update(self.config.baseline.parameters)
            # Add max-num-seqs parameter only if concurrency > 256
            if concurrency > 256:
                baseline_parameters["max_num_seqs"] = concurrency

            # Create special baseline trial configuration
            baseline_trial_config = TrialConfig(
                study_name=self.config.study_name,
                trial_id=f"baseline_concurrency_{concurrency}",
                trial_number=None,  # No Optuna trial number for baselines
                trial_type="baseline",
                parameters=baseline_parameters,
                parameter_configs=self.config.parameters,
                static_environment_variables=self.config.static_environment_variables,
                benchmark_config=self.config.benchmark,
                optimization_config=self.config.optimization,
                logging_config=self.config.logging_config,
            )

            try:
                # Submit baseline trial to execution backend
                job_handle = self.backend.submit_trial(baseline_trial_config)

                # Wait for baseline trial to complete using polling mechanism
                logger.info(
                    f"⏳ Waiting for baseline trial "
                    f"(concurrency={concurrency}) to complete..."
                )

                import time

                timeout_seconds = TWO_HOURS_IN_SECONDS
                start_time = time.time()
                poll_interval = POLL_RATE  # Poll every 5 seconds

                while time.time() - start_time < timeout_seconds:
                    # Poll for completion
                    completed_results, _ = self.backend.poll_trials([job_handle])

                    if completed_results:
                        # Trial completed
                        trial_result = completed_results[0]
                        logger.info(
                            f"✅ Baseline trial (concurrency={concurrency}) completed:"
                        )
                        if trial_result.success and trial_result.objective_values:
                            logger.info(
                                f"   Objectives: {trial_result.objective_values}"
                            )
                            # Store baseline results for comparison
                            self.baseline_results[concurrency] = (
                                trial_result.objective_values
                            )
                        else:
                            logger.error(
                                f"❌ Baseline trial failed: "
                                f"{trial_result.error_message}"
                            )
                        break
                    else:
                        # Still running, wait and poll again
                        time.sleep(poll_interval)
                else:
                    # Timeout reached
                    logger.error(
                        f"❌ Baseline trial (concurrency={concurrency}) "
                        f"timed out after {timeout_seconds} seconds"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Baseline trial (concurrency={concurrency}) failed: {e}"
                )
                # Continue with other concurrency levels
                continue

        logger.info("✅ Baseline trials completed")
