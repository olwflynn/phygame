import pytest
import pymunk
from src.game.physics import create_world


class TestPhysics:
    def test_create_world(self):
        gravity = (0, 900)
        space = create_world(gravity)
        
        assert isinstance(space, pymunk.Space)
        assert space.gravity == gravity
        assert not space.threaded  # We specified threaded=False

    def test_world_gravity(self):
        space = create_world((0, -500))
        assert space.gravity == (0, -500)

    def test_world_initialization(self):
        space = create_world((0, 0))
        assert len(space.shapes) == 0  # Should start empty
        # Note: static body exists but isn't in the bodies list by default
        assert space.static_body is not None
