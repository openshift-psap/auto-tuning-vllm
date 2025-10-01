from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


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

    min_value: Union[int, float] = Field(alias="min")
    max_value: Union[int, float] = Field(alias="max")
    step: Optional[Union[int, float]] = None
    data_type: str = "float"  # "int" or "float"

    def generate_optuna_suggest(self, trial) -> Union[int, float]:
        """Generate Optuna range suggestion."""
        if self.data_type == "int":
            return trial.suggest_int(
                self.name,
                int(self.min_value),
                int(self.max_value),
                step=int(self.step) if self.step else None,
            )
        else:
            return trial.suggest_float(
                self.name,
                float(self.min_value),
                float(self.max_value),
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
