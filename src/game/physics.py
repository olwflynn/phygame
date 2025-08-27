import pymunk
from typing import Tuple


def create_world(gravity: Tuple[float, float]) -> pymunk.Space:
    space = pymunk.Space(threaded=False)
    space.gravity = gravity
    return space
