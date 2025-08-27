import pytest
from src.game.ai import suggest_best_shot, ShotSuggestion


class TestAI:
    def test_shot_suggestion_structure(self):
        suggestion = suggest_best_shot()
        
        assert isinstance(suggestion, ShotSuggestion)
        assert hasattr(suggestion, 'angle_deg')
        assert hasattr(suggestion, 'force')
        assert isinstance(suggestion.angle_deg, float)
        assert isinstance(suggestion.force, float)

    def test_shot_suggestion_values(self):
        suggestion = suggest_best_shot()
        
        # Check that values are within reasonable bounds
        assert 0 <= suggestion.angle_deg <= 90
        assert suggestion.force > 0

    def test_shot_suggestion_dataclass(self):
        # Test creating a suggestion manually
        suggestion = ShotSuggestion(angle_deg=30.0, force=500.0)
        assert suggestion.angle_deg == 30.0
        assert suggestion.force == 500.0
