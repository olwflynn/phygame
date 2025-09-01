import pytest
import pymunk
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.entities import create_world, create_ground, create_target, create_bird


class TestEntities:
    def test_create_ground(self):
        space = pymunk.Space()
        ground = create_ground(space, y=500.0, width=960.0)
        
        assert isinstance(ground, pymunk.Segment)
        assert ground.friction == 0.98
        assert ground.elasticity == 0.3
        assert ground.body.body_type == pymunk.Body.STATIC

    def test_create_target(self):
        space = pymunk.Space()
        target = create_target(space, pos=(800.0, 400.0), size=(40.0, 40.0))
        
        assert isinstance(target, pymunk.Poly)
        assert target.friction == 0.9
        assert target.elasticity == 0.2
        assert target.body.body_type == pymunk.Body.DYNAMIC
        assert target.body.mass == 5.0

    def test_create_bird(self):
        space = pymunk.Space()
        bird = create_bird(space, pos=(120.0, 430.0), radius=14.0, velocity=(0.0, 0.0))
        
        assert isinstance(bird, pymunk.Circle)
        assert bird.friction == 0.95
        assert bird.elasticity == 0.4
        assert bird.body.body_type == pymunk.Body.DYNAMIC
        assert bird.body.mass == 1.0
        assert bird.radius == 14.0

    def test_create_bird_with_velocity(self):
        """Test creating bird with initial velocity"""
        space = pymunk.Space()
        bird = create_bird(space, pos=(120.0, 430.0), radius=14.0, velocity=(100.0, -50.0))
        
        assert bird.body.velocity.x == 100.0
        assert bird.body.velocity.y == -50.0

    def test_create_bird_default_velocity(self):
        """Test creating bird without velocity (should default to 0,0)"""
        space = pymunk.Space()
        bird = create_bird(space, pos=(120.0, 430.0), radius=14.0, velocity=(0.0, 0.0))
        
        assert bird.body.velocity.x == 0.0
        assert bird.body.velocity.y == 0.0
