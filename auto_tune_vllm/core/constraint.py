from dataclasses import dataclass
from typing import Any


@dataclass
class Constraint:
    """
    As per optuna, constraints are violated when the expression evaluates to > 0 and
    valid for values <= 0
    """

    expression: str

    def evaluate_constraint(self, parameters: dict[str, Any]) -> int | float:
        """
        Parse the expression string (e.g. max_num_batched_tokens - max_model_len) for
        all words. Then replace all words with the corresponding value in the dictionary
        parameters. Evaluate the expression and return the result otherwise error if the
        result is null.
        """
        try:
            return eval(self.expression, {"__builtins__": {}}, parameters)
        except Exception as error:
            raise error
