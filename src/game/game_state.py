import pymunk
from typing import Dict, Any, Optional, Tuple
from .entities import create_bird, create_target


def reset_bird(space: pymunk.Space, bird: pymunk.Circle, pos: Tuple[int, int]) -> pymunk.Circle:
    """Reset bird to starting position"""
    if bird is not None:
        space.remove(bird.body)
        space.remove(bird)
    bird = create_bird(space, pos=pos, radius=14, velocity=(0, 0))
    return bird


def reset_target(space: pymunk.Space, target: pymunk.Poly, pos: Tuple[int, int]) -> pymunk.Poly:
    """Reset target to starting position"""
    if target is not None:
        space.remove(target.body)
        space.remove(target)
    target = create_target(space, pos=pos, size=(40, 40))
    return target


def restart_simulation(space: pymunk.Space, bird: pymunk.Circle, target: pymunk.Poly, 
                      bird_start_pos: Tuple[int, int], target_start_pos: Tuple[int, int]) -> Tuple[pymunk.Circle, pymunk.Poly]:
    """Restart the simulation by resetting bird and target positions"""
    # Reset bird to starting position
    bird = reset_bird(space, bird, bird_start_pos)
    
    # Reset target to starting position
    target = reset_target(space, target, target_start_pos)
    
    return bird, target


def create_shot_data(shot_number: int, start_time: float, start_pos: Tuple[int, int]) -> Dict[str, Any]:
    """Create initial shot data dictionary"""
    return {
        'shot_number': shot_number,
        'start_time': start_time,
        'start_pos': start_pos
    }


def update_shot_data(shot_data: Dict[str, Any], end_pos: Tuple[int, int], 
                    velocity: Tuple[float, float], impulse_magnitude: float, 
                    angle_deg: float, drag_distance: float) -> Dict[str, Any]:
    """Update shot data with launch information"""
    shot_data.update({
        'end_pos': end_pos,
        'velocity': velocity,
        'impulse_magnitude': impulse_magnitude,
        'angle_deg': angle_deg,
        'drag_distance': drag_distance,
        'hit_target': False  # Will be updated when hit detection occurs
    })
    return shot_data


def finalize_shot_data(shot_data: Dict[str, Any], hit_target: bool) -> Dict[str, Any]:
    """Finalize shot data with hit result"""
    shot_data['hit_target'] = hit_target
    return shot_data
