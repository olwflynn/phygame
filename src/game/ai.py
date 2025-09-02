from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import math, random
import time
from . import physics
import matplotlib.pyplot as plt
from pymunk import Space, Body, Poly, Segment

from . import config
from . import entities

@dataclass
class ShotSuggestion:
    angle_deg: float
    impulse_magnitude: float

@dataclass
class SimulationState:
    running: bool = False
    progress: float = 0.0
    current_sample: int = 0
    total_samples: int = 1000
    results: list = None
    best_suggestion: Optional[ShotSuggestion] = None
    samples_per_frame: int = 10  # Run multiple samples per frame for better performance
    should_stop: bool = False  # Flag to immediately stop simulation

def _simulate_shot(space, target, angle_deg, impulse_magnitude, max_time=2.0):
    """
    Simulate a shot with given angle and impulse magnitude.
    Returns True if the bird hits the target bounding box.
    Uses the current space configuration including obstacles.
    Added timeout to prevent hanging.
    """
    start_time = time.time()
    
    # Create a new space for simulation with the same gravity
    sim_space = entities.create_world(space.gravity)
    
    # Copy the ground
    sim_ground = entities.create_ground(sim_space, y=500, width=960)
    
    # Copy the target
    sim_target = entities.create_target(sim_space, pos=target.body.position, size=(40, 40))
    
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
    
    # Let the target fall to its rest position
    target_rest_position = _let_target_fall_to_rest(sim_space, sim_target)
    
    # Create a simulation bird
    sim_bird = entities.create_bird(sim_space, pos=(120, 485), radius=14, velocity=(0,0))
    
    # Convert angle_deg and impulse_magnitude into velocity vector
    angle_rad = math.radians(angle_deg)
    vx = impulse_magnitude * math.cos(angle_rad)
    vy = -impulse_magnitude * math.sin(angle_rad)  # Negative because y increases downward in screen coordinates
    velocity = (vx, vy)
    sim_bird.body.apply_impulse_at_local_point(velocity)

    step_count = 0
    max_steps = 1800  # 30 seconds at 60 FPS
    
    while step_count < max_steps:
        # Check timeout
        if time.time() - start_time > max_time:
            return False  # Timeout - consider it a miss
        
        sim_space.step(1/60)
        step_count += 1

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
    
    return False  # Max steps reached - consider it a miss

def _let_target_fall_to_rest(space, target, max_steps=300):
    """
    Let the target fall until it reaches a rest state.
    Returns the final position of the target.
    
    Args:
        space: The physics space
        target: The target object
        max_steps: Maximum number of physics steps to wait for rest
    
    Returns:
        Tuple of (x, y) final position
    """
    rest_threshold = 5.0  # Velocity threshold for considering "at rest"
    consecutive_rest_steps = 0
    required_rest_steps = 5  # Number of consecutive steps at rest
    
    for step in range(max_steps):
        space.step(1/60)
        
        target_vel = target.body.velocity
        target_speed = math.sqrt(target_vel.x**2 + target_vel.y**2)
        
        if target_speed < rest_threshold:
            consecutive_rest_steps += 1
            if consecutive_rest_steps >= required_rest_steps:
                break
        else:
            consecutive_rest_steps = 0
    
    return target.body.position

def start_ai_simulation(space, target, sim_state: SimulationState, 
                       angle_min=0, angle_max=90, impulse_min=100, impulse_max=1200, 
                       N_SAMPLES=1000, plot=False):
    """
    Start AI simulation asynchronously. Updates sim_state with progress.
    """
    sim_state.running = True
    sim_state.progress = 0.0
    sim_state.current_sample = 0
    sim_state.total_samples = N_SAMPLES
    sim_state.results = []
    sim_state.best_suggestion = None
    sim_state.samples_per_frame = 10  # Run 10 samples per frame for better performance
    sim_state.should_stop = False  # Reset stop flag

def update_ai_simulation(space, target, sim_state: SimulationState, 
                        angle_min=0, angle_max=90, impulse_min=100, impulse_max=1200, 
                        plot=False):
    """
    Update AI simulation by running multiple samples per frame. Returns True if simulation is complete.
    """
    if not sim_state.running or sim_state.current_sample >= sim_state.total_samples or sim_state.should_stop:
        return True
    
    # Run multiple samples per frame for better performance
    samples_to_run = min(sim_state.samples_per_frame, sim_state.total_samples - sim_state.current_sample)
    
    for _ in range(samples_to_run):
        if sim_state.current_sample >= sim_state.total_samples or sim_state.should_stop:
            break
            
        # Run one simulation sample with timeout
        angle = random.uniform(angle_min, angle_max)
        impulse_magnitude = random.uniform(impulse_min, impulse_max)
        hit = _simulate_shot(space, target, angle, impulse_magnitude, max_time=2.0)
        
        sim_state.results.append((hit, angle, impulse_magnitude))
        sim_state.current_sample += 1
        sim_state.progress = sim_state.current_sample / sim_state.total_samples
    
    # Check if simulation is complete or stopped
    if sim_state.current_sample >= sim_state.total_samples or sim_state.should_stop:
        sim_state.running = False
        
        # Find best suggestion only if not stopped early
        if not sim_state.should_stop and sim_state.results:
            # Pick the last result where hit=True
            for hit, angle, impulse_magnitude in reversed(sim_state.results):
                if hit:
                    sim_state.best_suggestion = ShotSuggestion(angle_deg=angle, impulse_magnitude=impulse_magnitude)
                    break
        
        # Create plot if requested and not stopped early
        if not sim_state.should_stop and plot and sim_state.results:
            angles = [result[1] for result in sim_state.results]
            impulses = [result[2] for result in sim_state.results]
            hits = [result[0] for result in sim_state.results]
            
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
        
        return True
    
    return False

def stop_ai_simulation(sim_state: SimulationState):
    """Stop the AI simulation immediately"""
    sim_state.running = False
    sim_state.should_stop = True

# Legacy function for backward compatibility
def suggest_best_shot(space, target, angle_min=0, angle_max=90, impulse_min=100, impulse_max=1200, N_SAMPLES = 1000, plot=False):   
    """Legacy synchronous function - kept for backward compatibility"""
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
            space, target, angle, impulse_magnitude
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

    