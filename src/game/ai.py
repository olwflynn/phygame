from dataclasses import dataclass
from typing import Tuple


@dataclass
class ShotSuggestion:
    angle_deg: float
    force: float


def suggest_best_shot() -> ShotSuggestion:
    # Placeholder suggestion; will be replaced by Monte Carlo search
    return ShotSuggestion(angle_deg=45.0, force=600.0)
