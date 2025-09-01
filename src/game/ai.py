from dataclasses import dataclass
from typing import Tuple
import math, random
from . import physics
import matplotlib.pyplot as plt
from pymunk import Space, Body, Poly, Segment

from . import config
from . import entities

@dataclass
class ShotSuggestion:
    angle_deg: float
    impulse_magnitude: float

def _simulate_shot(space, angle_deg, impulse_magnitude):
    """
    Simulate a shot with given angle and impulse magnitude.
    Returns True if the bird hits the target bounding box.
    Uses the current space configuration including obstacles.
    """
    # Create a new space for simulation with the same gravity
    sim_space = entities.create_world(space.gravity)
    
    # Copy the ground
    sim_ground = entities.create_ground(sim_space, y=500, width=960)
    
    # Copy the target
    sim_target = entities.create_target(sim_space, pos=(800, 480), size=(40, 40))
    
    # Copy all obstacles from the original space
    for body in space.bodies:
        if body.body_type == Body.STATIC and body != space.static_body:
            # This is an obstacle
            for shape in body.shapes:
                if isinstance(shape, Poly):
                    # Create a copy of the obstacle
                    obstacle_body = Body(body_type=Body.STATIC)
                    obstacle_body.position = body.position
                # Copy the vertices directly from the original shape
                    vertices = shape.get_vertices()
                    obstacle_shape = Poly(obstacle_body, vertices)
                    obstacle_shape.friction = shape.friction
                    obstacle_shape.elasticity = shape.elasticity
                    sim_space.add(obstacle_body, obstacle_shape)
    
    # Create a simulation bird
    sim_bird = entities.create_bird(sim_space, pos=(120, 485), radius=14, velocity=(0,0))
    
    # Convert angle_deg and impulse_magnitude into velocity vector
    angle_rad = math.radians(angle_deg)
    vx = impulse_magnitude * math.cos(angle_rad)
    vy = -impulse_magnitude * math.sin(angle_rad)  # Negative because y increases downward in screen coordinates
    velocity = (vx, vy)
    sim_bird.body.apply_impulse_at_local_point(velocity)

    running = True
    while running:
        sim_space.step(1/60)

        bird_vel = sim_bird.body.velocity
        bird_pos = sim_bird.body.position

        # Check for target hit
        if entities.check_target_hit(sim_bird, sim_target):
            return True

         # Reset bird if it's moving very slowly (landed)
        if (physics.is_bird_landed((bird_vel.x, bird_vel.y)) and 
            bird_pos.x > 150):
            return False

        # Reset bird if it goes out of bounds
        if physics.is_bird_out_of_bounds((bird_pos.x, bird_pos.y)):
            return False
    
    return False

# Monte Carlo parameters

def suggest_best_shot(space, angle_min=0, angle_max=90, impulse_min=100, impulse_max=1200, N_SAMPLES = 1000, plot=False):   

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
            space, angle, impulse_magnitude
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

    