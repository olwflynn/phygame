import json
import pymunk

from .entities import create_target

def load_level(space):
    """Parse JSON and spawn objects into pymunk space"""
    data = json.loads(level_json)

    # --- Targets ---
    targets = []
    for t in data.get("targets", []):
        target = create_target(space, pos=(t["x"], t["y"]), size=(40, 40))
        targets.append(target)

    # --- Obstacles ---
    obstacles = []
    for o in data.get("obstacles", []):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (o["x"], o["y"])
        shape = pymunk.Poly.create_box(body, size=(o["w"], o["h"]))
        space.add(body, shape)
        obstacles.append((shape, (o["w"], o["h"])))

    return targets, obstacles


# --- Example usage ---
level_json = """
{
  "targets": [
    {"x": 800, "y": 400}
  ],
  "obstacles": [
    {"x": 400, "y": 500, "w": 10, "h": 100},
    {"x": 300, "y": 400, "w": 10, "h": 100},
    {"x": 700, "y": 500, "w": 10, "h": 100}
  ]
}
"""

