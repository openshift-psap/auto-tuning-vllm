from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import optuna
from pydantic import BaseModel, Field
from typing_extensions import override


class ParameterConfig(BaseModel, ABC):
    """Base class for parameter configurations."""

    name: str
    enabled: bool = True
    description: Optional[str] = None

    @abstractmethod
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna trial suggestion for this parameter."""
        pass


class RangeParameter(ParameterConfig):
    """Range-based parameter (continuous or discrete)."""

    min_value: int | float = Field(alias="min")
    max_value: int | float = Field(alias="max")
    step: int | float | None = None
    data_type: type[float | int] = float  # "int" or "float"

    @override
    def generate_optuna_suggest(self, trial: optuna.Trial) -> Union[int, float]:
        """Generate Optuna range suggestion."""
        if self.data_type is int:
            return trial.suggest_int(
                self.name,
                low=int(self.min_value),
                high=int(self.max_value),
                step=int(self.step) if self.step else 1,
            )
        else:
            return trial.suggest_float(
                self.name,
                low=float(self.min_value),
                high=float(self.max_value),
                step=float(self.step) if self.step else None,
            )


class ListParameter(ParameterConfig):
    """List-based parameter (categorical choices)."""

    options: List[Any]
    data_type: str = "str"

    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna categorical suggestion."""
        return trial.suggest_categorical(self.name, self.options)


class BooleanParameter(ParameterConfig):
    """Boolean parameter."""

    data_type: str = "bool"

    def generate_optuna_suggest(self, trial) -> bool:
        """Generate Optuna boolean suggestion."""
        return trial.suggest_categorical(self.name, [True, False])


class EnvironmentParameter(ParameterConfig):
    """Environment variable parameter (list-only choices)."""

    options: List[Any]
    data_type: str = "str"

    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna categorical suggestion for environment variable."""
        return trial.suggest_categorical(self.name, self.options)
