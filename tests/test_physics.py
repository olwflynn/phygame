import pytest
import pymunk
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.entities import create_world
from game.physics import calculate_launch_parameters, is_bird_landed, is_bird_out_of_bounds


class TestPhysics:
    def test_create_world(self):
        space = create_world((0, 900))
        
        assert isinstance(space, pymunk.Space)
        assert space.gravity == (0, 900)
    
    def test_world_gravity(self):
        space = create_world((0, 500))
        
        assert space.gravity == (0, 500)
    
    def test_world_initialization(self):
        space = create_world((100, 200))
        
        assert space.gravity == (100, 200)
        assert len(space.bodies) == 0
        assert len(space.shapes) == 0

    def test_calculate_launch_parameters(self):
        """Test launch parameter calculation"""
        start_pos = (120, 430)
        end_pos = (200, 400)
        velocity_multiplier = 5
        
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, velocity_multiplier
        )
        
        # Check return types
        assert isinstance(velocity, tuple)
        assert len(velocity) == 2
        assert isinstance(impulse_magnitude, (int, float))
        assert isinstance(angle_deg, (int, float))
        assert isinstance(drag_distance, (int, float))
        
        # Check velocity components
        assert isinstance(velocity[0], (int, float))
        assert isinstance(velocity[1], (int, float))
        
        # Check reasonable ranges
        assert impulse_magnitude > 0
        assert -180 <= angle_deg <= 180  # Angle in degrees (can be negative)
        assert drag_distance > 0

    def test_calculate_launch_parameters_horizontal_shot(self):
        """Test launch parameters for horizontal shot"""
        start_pos = (120, 430)
        end_pos = (220, 430)  # Same Y, different X
        velocity_multiplier = 5
        
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, velocity_multiplier
        )
        
        # Horizontal shot should have angle close to 0 degrees
        assert abs(angle_deg) < 10 or abs(angle_deg - 180) < 10 or abs(angle_deg + 180) < 10
        assert velocity[0] < 0  # Negative X velocity (drag right = launch left)
        assert abs(velocity[1]) < 50  # Small Y velocity

    def test_calculate_launch_parameters_vertical_shot(self):
        """Test launch parameters for vertical shot"""
        start_pos = (120, 430)
        end_pos = (120, 330)  # Same X, different Y (upward)
        velocity_multiplier = 5
        
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, velocity_multiplier
        )
        
        # Vertical shot should have angle close to 90 or -90 degrees
        assert abs(angle_deg - 90) < 10 or abs(angle_deg + 90) < 10
        assert abs(velocity[0]) < 50  # Small X velocity
        assert velocity[1] > 0  # Positive Y velocity (drag up = launch down)

    def test_calculate_launch_parameters_diagonal_shot(self):
        """Test launch parameters for diagonal shot"""
        start_pos = (120, 430)
        end_pos = (220, 330)  # Diagonal up and right
        velocity_multiplier = 5
        
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, velocity_multiplier
        )
        
        # Diagonal shot should have angle in reasonable range
        assert -180 < angle_deg < 180  # Any angle is valid
        assert velocity[0] < 0  # Negative X velocity (drag right = launch left)
        assert velocity[1] > 0  # Positive Y velocity (drag up = launch down)

    def test_calculate_launch_parameters_velocity_multiplier(self):
        """Test that velocity multiplier affects impulse magnitude"""
        start_pos = (120, 430)
        end_pos = (200, 400)
        
        # Test with different multipliers
        velocity1, impulse1, _, _ = calculate_launch_parameters(start_pos, end_pos, 5)
        velocity2, impulse2, _, _ = calculate_launch_parameters(start_pos, end_pos, 10)
        
        # Higher multiplier should result in higher impulse
        assert impulse2 > impulse1
        assert abs(velocity2[0]) > abs(velocity1[0])
        assert abs(velocity2[1]) > abs(velocity1[1])

    def test_is_bird_landed(self):
        """Test bird landed detection"""
        # Test with low velocity (should be landed)
        low_velocity = (2.0, 3.0)
        assert is_bird_landed(low_velocity) == True
        
        # Test with high velocity (should not be landed)
        high_velocity = (20.0, 15.0)
        assert is_bird_landed(high_velocity) == False
        
        # Test with zero velocity (should be landed)
        zero_velocity = (0.0, 0.0)
        assert is_bird_landed(zero_velocity) == True
        
        # Test with mixed velocities
        mixed_velocity = (2.0, 10.0)  # Low X, high Y
        assert is_bird_landed(mixed_velocity) == False
        
        mixed_velocity = (10.0, 2.0)  # High X, low Y
        assert is_bird_landed(mixed_velocity) == False

    def test_is_bird_landed_boundary_cases(self):
        """Test bird landed detection at boundary values"""
        # Test exactly at threshold (5.0) - should NOT be considered landed
        boundary_velocity = (5.0, 5.0)
        assert is_bird_landed(boundary_velocity) == False
        
        # Test just above threshold
        above_threshold = (5.1, 5.1)
        assert is_bird_landed(above_threshold) == False
        
        # Test just below threshold
        below_threshold = (4.9, 4.9)
        assert is_bird_landed(below_threshold) == True

    def test_is_bird_out_of_bounds(self):
        """Test bird out of bounds detection (only checks X coordinates)"""
        # Test positions within bounds (X coordinate)
        in_bounds_positions = [
            (100, 300),  # Left boundary
            (500, 300),  # Middle
            (960, 300),  # Right boundary
            (500, 100),  # Y doesn't matter
            (500, 500),  # Y doesn't matter
        ]
        
        for pos in in_bounds_positions:
            assert is_bird_out_of_bounds(pos) == False
        
        # Test positions out of bounds (X coordinate only)
        out_of_bounds_positions = [
            (1000, 300), # Too far right
            (-10, 300),  # Too far left
        ]
        
        for pos in out_of_bounds_positions:
            assert is_bird_out_of_bounds(pos) == True

    def test_is_bird_out_of_bounds_boundary_cases(self):
        """Test bird out of bounds detection at exact boundaries (X only)"""
        # Test exact boundaries (should be in bounds)
        boundary_positions = [
            (0, 300),    # Left boundary
            (960, 300),  # Right boundary
            (500, 100),  # Y doesn't matter
            (500, 540),  # Y doesn't matter
        ]
        
        for pos in boundary_positions:
            assert is_bird_out_of_bounds(pos) == False
        
        # Test just outside boundaries (X only)
        outside_positions = [
            (-1, 300),   # Just left of boundary
            (961, 300),  # Just right of boundary
        ]
        
        for pos in outside_positions:
            assert is_bird_out_of_bounds(pos) == True

    def test_physics_functions_with_different_data_types(self):
        """Test physics functions handle different data types correctly"""
        # Test with integers
        int_velocity = (3, 4)
        assert is_bird_landed(int_velocity) == True
        
        int_position = (500, 300)
        assert is_bird_out_of_bounds(int_position) == False
        
        # Test with floats
        float_velocity = (3.0, 4.0)
        assert is_bird_landed(float_velocity) == True
        
        float_position = (500.0, 300.0)
        assert is_bird_out_of_bounds(float_position) == False

    def test_launch_parameters_edge_cases(self):
        """Test launch parameter calculation with edge cases"""
        # Test with same start and end position
        same_pos = (120, 430)
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            same_pos, same_pos, 5
        )
        
        assert drag_distance == 0
        assert impulse_magnitude == 0
        assert velocity == (0, 0)
        
        # Test with very small movement
        small_end = (121, 431)
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            same_pos, small_end, 5
        )
        
        assert drag_distance > 0
        assert impulse_magnitude > 0
        assert velocity != (0, 0)
