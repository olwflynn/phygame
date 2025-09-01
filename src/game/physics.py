import math
from typing import Tuple


def calculate_launch_parameters(start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                               velocity_multiplier: float) -> Tuple[Tuple[float, float], float, float, float]:
    """
    Calculate launch parameters from drag start and end positions
    
    Returns:
        velocity: Tuple of (x_velocity, y_velocity)
        impulse_magnitude: Magnitude of the impulse
        angle_deg: Launch angle in degrees
        drag_distance: Distance of the drag
    """
    # Calculate launch direction: from start to end (opposite of current)
    dx = start_pos[0] - end_pos[0]  # Reverse X direction (left drag = right launch)
    dy = start_pos[1] - end_pos[1]  # Reverse Y direction (up drag = down launch)
    velocity = (dx * velocity_multiplier, dy * velocity_multiplier)
    
    # Calculate shot parameters
    drag_distance = math.sqrt(dx**2 + dy**2) / 2
    impulse_magnitude = drag_distance * velocity_multiplier
    angle_rad = math.atan2(-dy, dx)  # Calculate angle (negative dy because y increases downward)
    angle_deg = math.degrees(angle_rad)
    
    return velocity, impulse_magnitude, angle_deg, drag_distance


def is_bird_landed(bird_velocity: Tuple[float, float], threshold: float = 5.0) -> bool:
    """Check if bird has landed (moving very slowly)"""
    return abs(bird_velocity[0]) < threshold and abs(bird_velocity[1]) < threshold


def is_bird_out_of_bounds(bird_position: Tuple[float, float], width: float = 960) -> bool:
    """Check if bird is out of bounds"""
    return bird_position[0] > width or bird_position[0] < 0
