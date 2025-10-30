"""Trial logger management for vLLM auto-tuning trials."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.trial import TrialConfig

from ...logging.manager import CentralizedLogger

logger = logging.getLogger(__name__)


class TrialLoggers:
    """Manages trial-specific loggers for controller, vLLM, and benchmark components."""

    def __init__(self):
        self.trial_loggers: dict[str, logging.Logger] = {}

    def setup(self, trial_config: TrialConfig) -> None:
        """Setup trial-specific loggers based on logging configuration."""
        if not trial_config.logging_config:
            # No specific logging config, use default loggers
            return

        try:
            # Initialize CentralizedLogger for this trial
            log_database_url = trial_config.logging_config.get("database_url")
            log_file_path = trial_config.logging_config.get("file_path")
            log_level = trial_config.logging_config.get("log_level", "INFO")

            if log_database_url or log_file_path:
                centralized_logger = CentralizedLogger(
                    study_name=trial_config.study_name,
                    pg_url=log_database_url,
                    file_path=log_file_path,
                    log_level=log_level,
                )

                # Get trial-specific loggers for different components
                self.trial_loggers["controller"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "controller"
                )
                self.trial_loggers["vllm"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "vllm"
                )
                self.trial_loggers["benchmark"] = centralized_logger.get_trial_logger(
                    trial_config.trial_id, "benchmark"
                )

                # Log trial start
                self.trial_loggers["controller"].info(
                    f"Starting trial {trial_config.trial_id}"
                )
                self.trial_loggers["controller"].info(
                    f"Parameters: {trial_config.parameters}"
                )

        except Exception as e:
            # Fallback to default logger if setup fails
            logger.warning(f"Failed to setup trial logging: {e}")

    def get_logger(self, component: str) -> logging.Logger:
        """
        Get trial logger for specific component.

        Fallback to default if not available.
        """
        return self.trial_loggers.get(component, logger)

    def flush_handlers(self, target_logger: logging.Logger) -> None:
        """Immediately flush all handlers for a specific logger."""
        for handler in target_logger.handlers:
            try:
                handler.flush()
            except Exception as e:
                # Silently ignore flush errors to avoid breaking cleanup
                logger.debug(f"Failed to flush handler: {e}")

    def flush_all(self, trial_id: str, study_name: str | None = None) -> None:
        """Flush any buffered logs for the trial to ensure all records are written."""
        try:
            # Flush trial-specific loggers if we have them
            for component_logger in self.trial_loggers.values():
                for handler in component_logger.handlers:
                    try:
                        handler.flush()
                    except Exception as e:
                        logger.debug(f"Failed to flush handler: {e}")

            # Also try to flush by logger name pattern (fallback)
            if study_name:
                for component in ["controller", "vllm", "benchmark"]:
                    logger_name = f"study_{study_name}.{trial_id}.{component}"
                    trial_logger = logging.getLogger(logger_name)
                    for handler in trial_logger.handlers:
                        try:
                            handler.flush()
                        except Exception as e:
                            logger.debug(
                                f"Failed to flush handler for {logger_name}: {e}"
                            )
        except Exception as e:
            logger.debug(f"Error flushing trial logs: {e}")
