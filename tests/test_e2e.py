import pytest
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestE2E:
    """End-to-end tests that test the full game flow"""
    
    def test_game_initialization(self):
        """Test that the game can be initialized without errors"""
        # This test just ensures we can import the main module
        # and that basic game objects can be created
        from game.physics import create_world
        from game.entities import create_ground, create_target
        
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        
        assert len(space.bodies) == 1  # only target (ground uses static_body)
    
    # def test_ai_suggestion(self):
    #     """Test that AI can provide shot suggestions"""
    #     from game.ai import suggest_best_shot
        
    #     suggestion = suggest_best_shot()
    #     assert suggestion.angle_deg == 45.0  # Current stub value
    #     assert suggestion.force == 600.0     # Current stub value
