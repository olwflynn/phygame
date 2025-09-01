from dataclasses import dataclass
from typing import Tuple
import math, random
import matplotlib.pyplot as plt

from . import config
from . import entities

@dataclass
class ShotSuggestion:
    angle_deg: float
    impulse_magnitude: float

def _simulate_shot(angle_deg, impulse_magnitude):
    """
    Simulate a shot with given angle and impulse magnitude.
    Returns True if the bird hits the target bounding box.
    """
     # Create physics world and entities
    space = entities.create_world((0, 900))  # Gravity: 900 pixels/s² downward
    ground = entities.create_ground(space, y=500, width=960)  # Target starts on the ground
    target = entities.create_target(space, pos=(800, 480), size=(40, 40))  # Target starts on the ground
    bird = entities.create_bird(space, pos=(120, 485), radius=14, velocity=(0,0))  # Bird starts on the ground
    
    
    # Convert angle_deg and impulse_magnitude into velocity vector
    angle_rad = math.radians(angle_deg)
    vx = impulse_magnitude * math.cos(angle_rad)
    vy = -impulse_magnitude * math.sin(angle_rad)  # Negative because y increases downward in screen coordinates
    velocity = (vx, vy)
    bird.body.apply_impulse_at_local_point(velocity)

    running = True
    while running:
    
        space.step(1/60)

        # Check for target hit
        if entities.check_target_hit(bird, target):
            return True

        # Reset bird if it's moving very slowly (landed)
        if (abs(bird.body.velocity.x) < 5 and 
            abs(bird.body.velocity.y) < 5 and 
            bird.body.position.x > 500):
            return False

        if (abs(bird.body.velocity.x) < 1 and   
            abs(bird.body.velocity.y) < 1):
                    return False

        if bird.body.position.x > 960 or bird.body.position.x < 0:
            return False

        if bird.body.position.y > 900:
            return False

        if bird.body.position.y < 0:
            return False
    
    return False

# Monte Carlo parameters

def suggest_best_shot(angle_min=10, angle_max=80, impulse_min=100, impulse_max=1200, N_SAMPLES = 200, plot=False):   

    results = []

    for _ in range(N_SAMPLES):
        # Simple progress bar
        bar_length = 30
        progress = (_ + 1) / N_SAMPLES
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rSimulation Progress: |{bar}| {(_ + 1)}/{N_SAMPLES}", end='', flush=True)
        if _ + 1 == N_SAMPLES:
            print()  # Newline at end
        angle = random.uniform(angle_min, angle_max)
        impulse_magnitude = random.uniform(impulse_min, impulse_max)
        hit = _simulate_shot(
            angle, impulse_magnitude
        )
        results.append((hit, angle, impulse_magnitude))
    
    # Create scatter plot if requested
    if plot and results:
        angles = [result[1] for result in results]
        impulses = [result[2] for result in results]
        hits = [result[0] for result in results]
        
        # Separate hits and misses
        hit_angles = [angle for angle, hit in zip(angles, hits) if hit]
        hit_impulses = [impulse for impulse, hit in zip(impulses, hits) if hit]
        miss_angles = [angle for angle, hit in zip(angles, hits) if not hit]
        miss_impulses = [impulse for impulse, hit in zip(impulses, hits) if not hit]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(hit_angles, hit_impulses, color='blue', alpha=0.6, label='Hit', s=30)
        plt.scatter(miss_angles, miss_impulses, color='red', alpha=0.6, label='Miss', s=30)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Impulse Magnitude')
        plt.title('Shot Results: Angle vs Impulse Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
            
    if results:
        # Pick the last result where hit=True
        for hit, angle, impulse_magnitude in reversed(results):
            if hit:
                best_angle = angle
                best_impulse_magnitude = impulse_magnitude
                return ShotSuggestion(angle_deg=best_angle, impulse_magnitude=best_impulse_magnitude)
        else:
            print("No result found")
            return None

    