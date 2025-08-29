from dataclasses import dataclass
from typing import Tuple
import math, random
from . import config

@dataclass
class ShotSuggestion:
    angle_deg: float
    force: float

def _simulate_shot(angle_deg, force, start_pos, target_pos, target_size, bird_radius, gravity, dt=1/60.0, max_time=3.0):
    """
    Simulate a shot with given angle and force.
    Returns True if the bird hits the target bounding box.
    """
    angle_rad = math.radians(angle_deg)
    vx = force * math.cos(angle_rad)
    vy = -force * math.sin(angle_rad)  # Negative because y increases downward

    x, y = start_pos
    t = 0.0

    target_x, target_y = target_pos
    target_w, target_h = target_size

    # Target bounding box
    left = target_x - target_w / 2 - bird_radius
    right = target_x + target_w / 2 + bird_radius
    top = target_y - target_h / 2 - bird_radius
    bottom = target_y + target_h / 2 + bird_radius

    while t < max_time:
        # Update position
        x += vx * dt
        y += vy * dt
        vy += gravity * dt
        t += dt

        # Check if bird is within target bounds
        if left <= x <= right and top <= y <= bottom:
            return True

        # If bird falls below ground, stop
        if y > config.WINDOW_HEIGHT:
            break

    return False

# Monte Carlo parameters
N_SAMPLES = 200

def suggest_best_shot() -> ShotSuggestion:
    best_score = -1
    best_angle = None
    best_force = None

    # Use config values
    start_pos = (config.BIRD_START_X, config.BIRD_START_Y)
    target_pos = (config.TARGET_X, config.TARGET_Y)
    target_size = (config.TARGET_WIDTH, config.TARGET_HEIGHT)
    bird_radius = config.BIRD_RADIUS
    gravity = config.GRAVITY

    # Reasonable ranges for angle and force
    angle_min, angle_max = 10, 80
    force_min, force_max = 300, 1200

    results = []

    for _ in range(N_SAMPLES):
        angle = random.uniform(angle_min, angle_max)
        force = random.uniform(force_min, force_max)
        hit = _simulate_shot(
            angle, force, start_pos, target_pos, target_size, bird_radius, gravity
        )
        if hit:
            # Prefer lower force (more efficient) and angle closer to 45
            score = 1000 - abs(angle - 45) - 0.1 * force
            results.append((score, angle, force))

    if results:
        # Pick the best scoring shot
        results.sort(reverse=True)
        _, best_angle, best_force = results[0]
    else:
        # Fallback to default
        best_angle = 45.0
        best_force = 600.0

    return ShotSuggestion(angle_deg=best_angle, force=best_force)
      
# print(suggest_best_shot())