import pytest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestE2E:
    """End-to-end tests that test the full game flow"""
    
    def test_game_initialization(self):
        """Test that the game can be initialized without errors"""
        # This test just ensures we can import the main module
        # and that basic game objects can be created
        from game.entities import create_world
        from game.entities import create_ground, create_target, create_bird
        
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
        
        assert len(space.bodies) == 2  # target and bird (ground uses static_body)
    
    def test_ai_suggestion_integration(self):
        """Test that AI can provide shot suggestions in a real scenario"""
        from game.entities import create_world, create_ground, create_target, create_bird
        from game.ai import suggest_best_shot
        
        # Set up a real game scenario
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
        
        # Test AI suggestion with mocked simulation for speed
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            
            suggestion = suggest_best_shot(space, target, N_SAMPLES=5, plot=False)
            
            assert suggestion is not None
            assert hasattr(suggestion, 'angle_deg')
            assert hasattr(suggestion, 'impulse_magnitude')
            assert isinstance(suggestion.angle_deg, (int, float))
            assert isinstance(suggestion.impulse_magnitude, (int, float))
    
    def test_shot_tracking_workflow(self):
        """Test complete shot tracking workflow"""
        from game.game_state import create_shot_data, update_shot_data, finalize_shot_data
        from game.physics import calculate_launch_parameters
        
        # Step 1: Create shot data
        shot_data = create_shot_data(1, time.time(), (120, 430))
        assert shot_data['shot_number'] == 1
        
        # Step 2: Calculate launch parameters
        start_pos = (120, 430)
        end_pos = (200, 400)
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, 5
        )
        
        # Step 3: Update shot data
        shot_data = update_shot_data(
            shot_data, end_pos, velocity, impulse_magnitude, angle_deg, drag_distance
        )
        assert shot_data['velocity'] == velocity
        assert shot_data['angle_deg'] == angle_deg
        
        # Step 4: Finalize shot data (simulate hit)
        shot_data = finalize_shot_data(shot_data, True)
        assert shot_data['hit_target'] == True
        
        # Verify complete workflow
        assert shot_data['shot_number'] == 1
        assert shot_data['start_pos'] == start_pos
        assert shot_data['end_pos'] == end_pos
        assert shot_data['hit_target'] == True
    
    def test_physics_integration(self):
        """Test physics functions work together"""
        from game.entities import create_world, create_ground, create_bird
        from game.physics import calculate_launch_parameters, is_bird_landed, is_bird_out_of_bounds
        
        # Set up physics world
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
        
        # Test launch parameter calculation
        start_pos = (120, 430)
        end_pos = (200, 400)
        velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
            start_pos, end_pos, 5
        )
        
        # Apply velocity to bird
        bird.body.velocity = velocity
        
        # Test physics state detection
        assert not is_bird_landed(velocity)  # Bird should be moving
        assert not is_bird_out_of_bounds((120, 430))  # Bird should be in bounds
        
        # Test with landed state
        bird.body.velocity = (2, 3)  # Low velocity
        assert is_bird_landed((2, 3))  # Bird should be considered landed
        
        # Test with out of bounds (X coordinate only)
        assert is_bird_out_of_bounds((1000, 300))  # Should be out of bounds
    
    def test_ui_components_integration(self):
        """Test UI components work together"""
        from game.ui import create_shot_table, update_charts
        from collections import deque
        
        # Test shot table creation
        shot_history = [
            {
                'shot_number': 1,
                'angle_deg': 45.0,
                'impulse_magnitude': 500.0,
                'hit_target': True
            }
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = create_shot_table(shot_history)
            assert result == mock_fig
        
        # Test chart update
        time_data = deque([0.0, 0.1, 0.2], maxlen=300)
        x_pos_data = deque([120, 150, 180], maxlen=300)
        y_pos_data = deque([430, 420, 410], maxlen=300)
        x_vel_data = deque([300, 280, 260], maxlen=300)
        y_vel_data = deque([-200, -180, -160], maxlen=300)
        
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        
        # Should not raise exceptions
        update_charts(
            mock_fig, mock_ax1, mock_ax2, mock_ax3,
            time_data, x_pos_data, y_pos_data, x_vel_data, y_vel_data
        )
    
    def test_config_integration(self):
        """Test configuration constants are used correctly"""
        from game import config
        from game.entities import create_world, create_bird
        
        # Test that config values are used in entity creation
        space = create_world(config.GRAVITY)
        bird = create_bird(space, pos=config.SLINGSHOT_POS, radius=config.BIRD_RADIUS, velocity=(0, 0))
        
        assert space.gravity == config.GRAVITY
        assert bird.radius == config.BIRD_RADIUS
        assert bird.body.position == config.SLINGSHOT_POS
    
    def test_game_state_management(self):
        """Test game state management functions work together"""
        from game.entities import create_world, create_ground, create_target, create_bird
        from game.game_state import reset_bird, reset_target, create_shot_data, update_shot_data, finalize_shot_data
        
        # Set up game state
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
        
        # Test reset functions
        bird.body.position = (500, 300)
        target.body.position = (600, 300)
        
        new_bird = reset_bird(space, bird, (120, 430))
        new_target = reset_target(space, target, (800, 400))
        
        assert new_bird.body.position == (120, 430)
        assert new_target.body.position == (800, 400)
        
        # Test shot data management
        shot_data = create_shot_data(1, time.time(), (120, 430))
        shot_data = update_shot_data(shot_data, (200, 400), (150, -200), 250, 45, 100)
        shot_data = finalize_shot_data(shot_data, True)
        
        assert shot_data['hit_target'] == True
        assert shot_data['shot_number'] == 1
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        from game.game_state import create_shot_data, update_shot_data, finalize_shot_data
        
        # Test with invalid data types
        shot_data = create_shot_data(1, time.time(), (120, 430))
        
        # Should handle updates gracefully
        shot_data = update_shot_data(shot_data, (200, 400), (150, -200), 250, 45, 100)
        shot_data = finalize_shot_data(shot_data, False)
        
        assert shot_data['hit_target'] == False
        
        # Test UI error handling
        from game.ui import create_shot_table
        
        # Should handle empty data gracefully
        assert create_shot_table([]) is None
        assert create_shot_table(None) is None
    
    def test_performance_considerations(self):
        """Test that functions perform reasonably"""
        from game.entities import create_world, create_ground, create_target, create_bird
        from game.physics import calculate_launch_parameters
        import time
        
        # Set up game
        space = create_world((0, 900))
        ground = create_ground(space, y=500, width=960)
        target = create_target(space, pos=(800, 400), size=(40, 40))
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
        
        # Test launch parameter calculation performance
        start_time = time.time()
        for _ in range(100):
            calculate_launch_parameters((120, 430), (200, 400), 5)
        end_time = time.time()
        
        # Should complete 100 calculations quickly (less than 1 second)
        assert (end_time - start_time) < 1.0
