import pytest
import pygame
import os
from src.main import main


class TestE2E:
    def test_game_imports(self):
        """Test that all game modules can be imported without errors"""
        import src.game.config
        import src.game.entities
        import src.game.physics
        import src.game.ai
        
        # Check that config has expected constants
        assert src.game.config.WINDOW_WIDTH == 960
        assert src.game.config.WINDOW_HEIGHT == 540
        assert src.game.config.FPS == 60

    def test_physics_world_creation(self):
        """Test that physics world can be created and stepped"""
        from src.game.physics import create_world
        from src.game.entities import create_ground, create_target
        
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        
        # Step physics a few times
        for _ in range(10):
            space.step(1/60)
        
        # Check that entities still exist
        assert len(space.shapes) == 2  # ground + target
        assert len(space.bodies) == 1  # only target (ground uses static_body)

    def test_ai_suggestion(self):
        """Test that AI can provide shot suggestions"""
        from src.game.ai import suggest_best_shot
        
        suggestion = suggest_best_shot()
        assert suggestion.angle_deg == 45.0  # Current stub value
        assert suggestion.force == 600.0     # Current stub value
