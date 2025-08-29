import pytest
import pymunk
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.physics import create_world


class TestPhysics:
    def test_create_world(self):
        space = create_world((0, 900))
        
        assert isinstance(space, pymunk.Space)
        assert space.gravity == (0, 900)
    
    def test_world_gravity(self):
        space = create_world((0, 500))
        
        assert space.gravity == (0, 500)
    
    def test_world_initialization(self):
        space = create_world((100, 200))
        
        assert space.gravity == (100, 200)
        assert len(space.bodies) == 0
        assert len(space.shapes) == 0
