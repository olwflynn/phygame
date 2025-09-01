from typing import Tuple
import pymunk


def create_world(gravity: Tuple[float, float]) -> pymunk.Space:
    space = pymunk.Space(threaded=False)
    space.gravity = gravity
    return space

def create_ground(space: pymunk.Space, y: float, width: float) -> pymunk.Segment:
    static_body = space.static_body
    segment = pymunk.Segment(static_body, (0, y), (width, y), 1.0)
    segment.friction = 0.98  # Increased friction
    segment.elasticity = 0.3
    space.add(segment)
    return segment


def create_target(space: pymunk.Space, pos: Tuple[float, float], size: Tuple[float, float]) -> pymunk.Poly:
    body = pymunk.Body(5, pymunk.moment_for_box(5, size))
    body.position = pos
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = 0.9
    shape.elasticity = 0.2
    space.add(body, shape)
    return shape


def create_bird(space: pymunk.Space, pos: Tuple[float, float], radius: float, velocity: Tuple[float, float]) -> pymunk.Circle:
    mass = 1.0
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    body.velocity = velocity  # Actually set the velocity
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.95
    shape.elasticity = 0.4
    space.add(body, shape)
    return shape


def check_target_hit(bird, target):
    """Check if bird hit the target"""
    bird_pos = bird.body.position
    target_pos = target.body.position
    
    # Simple distance-based collision detection
    distance = ((bird_pos.x - target_pos.x)**2 + (bird_pos.y - target_pos.y)**2)**0.5
    return distance < 35  # Bird radius + half target size
