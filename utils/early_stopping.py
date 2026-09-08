"""Early stopping on a monitored validation metric."""

from typing import Literal


class EarlyStopping:
    """
    Stops training when a monitored metric has not improved for `patience` checks.

    Call with the latest metric value each validation step; returns True when
    training should stop.
    """

    def __init__(
        self,
        patience: int = 15,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.num_bad_checks = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def __call__(self, value: float) -> bool:
        if self._is_improvement(value):
            self.best = value
            self.num_bad_checks = 0
            return False
        self.num_bad_checks += 1
        return self.num_bad_checks >= self.patience
