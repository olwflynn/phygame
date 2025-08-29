import pytest
import sys
import os

# # Add src directory to Python path so we can import from main.py
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# from game.ai import suggest_best_shot, ShotSuggestion


# class TestAI:
#     def test_shot_suggestion_structure(self):
#         """Test that AI suggestion returns proper structure"""
#         suggestion = suggest_best_shot()
        
#         assert isinstance(suggestion, ShotSuggestion)
#         assert suggestion.angle_deg == 45.0
#         assert suggestion.force == 600.0
