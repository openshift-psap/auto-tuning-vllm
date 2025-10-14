from enum import Enum
from pathlib import Path

import optuna

from ..config import StudyConfig


class StorageType(Enum):
    IN_MEMORY = 0
    SQLITE = 1
    POSTGRESQL = 2


def get_storage(
    config: StudyConfig,
    resume_study: bool = False,
) -> tuple[optuna.storages.BaseStorage, StorageType]:
    # Determine storage backend for Optuna study
    retry_callback = optuna.storages.RetryFailedTrialCallback(max_retry=0)
    if config.database_url:
        return optuna.storages.RDBStorage(
            url=config.database_url, failed_trial_callback=retry_callback
        ), StorageType.POSTGRESQL
    elif config.storage_file:
        # Ensure directory exists for file-based storage
        storage_path = Path(config.storage_file)
        if resume_study and not storage_path.exists():
            raise RuntimeError(
                f"Study '{config.study_name}' not found in storage. "
                f"Storage file does not exist: {config.storage_file}. "
                f"Options:\n"
                f"  • Use 'auto-tune-vllm optimize --config <config_file>' "
                f"to create a new study\n"
                f"  • Verify the study name and storage path are correct"
            )
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        return optuna.storages.RDBStorage(
            url=f"sqlite:///{config.storage_file}",
            failed_trial_callback=retry_callback,
        ), StorageType.SQLITE

    # This should not happen due to validation in config.py, but fallback to in-memory
    return optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        failed_trial_callback=retry_callback,
    ), StorageType.IN_MEMORY
