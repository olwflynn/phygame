import pytest
import pymunk
from src.game.entities import create_ground, create_target, create_bird


class TestEntities:
    def test_create_ground(self):
        space = pymunk.Space()
        ground = create_ground(space, y=500.0, width=960.0)
        
        assert isinstance(ground, pymunk.Segment)
        assert ground.friction == 0.9
        assert ground.elasticity == 0.3
        assert ground.body.body_type == pymunk.Body.STATIC

    def test_create_target(self):
        space = pymunk.Space()
        target = create_target(space, pos=(800.0, 400.0), size=(40.0, 40.0))
        
        assert isinstance(target, pymunk.Poly)
        assert target.friction == 0.6
        assert target.elasticity == 0.2
        assert target.body.body_type == pymunk.Body.DYNAMIC
        assert target.body.mass == 5.0

    def test_create_bird(self):
        space = pymunk.Space()
        bird = create_bird(space, pos=(120.0, 430.0), radius=14.0)
        
        assert isinstance(bird, pymunk.Circle)
        assert bird.friction == 0.6
        assert bird.elasticity == 0.4
        assert bird.body.body_type == pymunk.Body.DYNAMIC
        assert bird.body.mass == 1.0
        assert bird.radius == 14.0
