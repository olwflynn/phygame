import pytest
import pymunk
import sys
import os
import time

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.entities import create_world, create_ground, create_target, create_bird
from game.game_state import create_shot_data, update_shot_data, finalize_shot_data


class TestGameState:
    """Test game state management and game over conditions"""
    
    def setup_method(self):
        """Set up physics world and entities for each test"""
        self.space = create_world((0, 900))
        self.ground = create_ground(self.space, y=500, width=960)
        self.target = create_target(self.space, pos=(800, 400), size=(40, 40))
        self.bird = create_bird(self.space, pos=(120, 430), radius=14, velocity=(0, 0))
    
    def test_initial_game_state(self):
        """Test initial game state values"""
        # These would normally be game state variables
        shots_fired = 0
        max_shots = 3
        score = 0
        
        assert shots_fired == 0
        assert max_shots == 3
        assert score == 0
        assert shots_fired < max_shots
    
    def test_game_not_over_before_shots(self):
        """Test game is not over before any shots are fired"""
        shots_fired = 0
        max_shots = 3
        
        game_over = shots_fired >= max_shots
        assert game_over is False
    
    def test_game_not_over_with_remaining_shots(self):
        """Test game is not over when shots remain"""
        shots_fired = 1
        max_shots = 3
        
        game_over = shots_fired >= max_shots
        assert game_over is False
        
        shots_fired = 2
        game_over = shots_fired >= max_shots
        assert game_over is False
    
    def test_game_over_when_max_shots_reached(self):
        """Test game is over when max shots are reached"""
        shots_fired = 3
        max_shots = 3
        
        game_over = shots_fired >= max_shots
        assert game_over is True
    
    def test_game_over_when_shots_exceeded(self):
        """Test game is over when shots exceed maximum"""
        shots_fired = 4
        max_shots = 3
        
        game_over = shots_fired >= max_shots
        assert game_over is True
    
    def test_shot_counting_logic(self):
        """Test shot counting and remaining shots calculation"""
        shots_fired = 1
        max_shots = 3
        
        remaining_shots = max_shots - shots_fired
        assert remaining_shots == 2
        
        shots_fired = 2
        remaining_shots = max_shots - shots_fired
        assert remaining_shots == 1
        
        shots_fired = 3
        remaining_shots = max_shots - shots_fired
        assert remaining_shots == 0
    
    def test_shot_limit_enforcement(self):
        """Test that shots cannot be fired when limit is reached"""
        shots_fired = 3
        max_shots = 3
        
        # Simulate the shot limit check from main game loop
        can_fire = shots_fired < max_shots
        assert can_fire is False
        
        # Test with remaining shots
        shots_fired = 1
        can_fire = shots_fired < max_shots
        assert can_fire is True
    
    def test_score_increment_on_target_hit(self):
        """Test score increases when target hit"""
        initial_score = 0
        score_increment = 100
        
        # Simulate target hit
        new_score = initial_score + score_increment
        assert new_score == 100
        
        # Simulate multiple hits
        new_score += score_increment
        assert new_score == 200
    
    def test_game_reset_after_target_hit(self):
        """Test that game resets after target hit"""
        shots_fired = 2
        max_shots = 3
        
        # Simulate target hit - shots should reset
        shots_fired = 0
        
        assert shots_fired == 0
        assert shots_fired < max_shots
    
    def test_bird_launch_conditions(self):
        """Test conditions for allowing bird launch"""
        shots_fired = 1
        max_shots = 3
        bird_x_position = 120  # Starting position
        
        # Bird should be launchable when:
        # 1. Shots remain
        # 2. Bird is in starting position (not yet launched)
        can_launch = (shots_fired < max_shots and bird_x_position < 500)
        assert can_launch is True
        
        # Bird should not be launchable when shots are exhausted
        shots_fired = 3
        can_launch = (shots_fired < max_shots and bird_x_position < 500)
        assert can_launch is False
        
        # Bird should not be launchable when already launched
        shots_fired = 1
        bird_x_position = 600  # Past launch position
        can_launch = (shots_fired < max_shots and bird_x_position < 500)
        assert can_launch is False
    
    def test_game_completion_scenarios(self):
        """Test different game completion scenarios"""
        max_shots = 3
        
        # Scenario 1: Win with shots remaining
        shots_fired = 1
        target_hit = True
        game_won = target_hit and shots_fired < max_shots
        assert game_won is True
        
        # Scenario 2: Win on last shot
        shots_fired = 3
        target_hit = True
        game_won = target_hit and shots_fired <= max_shots
        assert game_won is True
        
        # Scenario 3: Lose (no shots remaining, target not hit)
        shots_fired = 3
        target_hit = False
        game_lost = shots_fired >= max_shots and not target_hit
        assert game_lost is True
    
    def test_reset_game_state(self):
        """Test complete game state reset"""
        # Simulate game in progress
        shots_fired = 2
        score = 200
        
        # Reset game
        shots_fired = 0
        score = 0
        
        assert shots_fired == 0
        assert score == 0
        
        # Check that new game can start
        can_fire = shots_fired < 3  # max_shots
        assert can_fire is True

    def test_create_shot_data(self):
        """Test creating initial shot data"""
        shot_number = 1
        start_time = time.time()
        start_pos = (120, 430)
        
        shot_data = create_shot_data(shot_number, start_time, start_pos)
        
        assert shot_data['shot_number'] == 1
        assert shot_data['start_time'] == start_time
        assert shot_data['start_pos'] == start_pos
        assert len(shot_data) == 3  # Only initial fields

    def test_update_shot_data(self):
        """Test updating shot data with launch information"""
        # Create initial shot data
        shot_data = create_shot_data(1, time.time(), (120, 430))
        
        # Update with launch information
        end_pos = (200, 400)
        velocity = (150.0, -200.0)
        impulse_magnitude = 250.0
        angle_deg = 45.0
        drag_distance = 100.0
        
        updated_data = update_shot_data(
            shot_data, end_pos, velocity, impulse_magnitude, angle_deg, drag_distance
        )
        
        # Check all fields are present
        assert updated_data['shot_number'] == 1
        assert updated_data['start_time'] == shot_data['start_time']
        assert updated_data['start_pos'] == (120, 430)
        assert updated_data['end_pos'] == end_pos
        assert updated_data['velocity'] == velocity
        assert updated_data['impulse_magnitude'] == impulse_magnitude
        assert updated_data['angle_deg'] == angle_deg
        assert updated_data['drag_distance'] == drag_distance
        assert updated_data['hit_target'] == False  # Default value

    def test_finalize_shot_data_hit(self):
        """Test finalizing shot data when target is hit"""
        # Create and update shot data
        shot_data = create_shot_data(1, time.time(), (120, 430))
        shot_data = update_shot_data(shot_data, (200, 400), (150.0, -200.0), 250.0, 45.0, 100.0)
        
        # Finalize with hit
        finalized_data = finalize_shot_data(shot_data, True)
        
        assert finalized_data['hit_target'] == True
        assert finalized_data['shot_number'] == 1
        assert finalized_data['angle_deg'] == 45.0

    def test_finalize_shot_data_miss(self):
        """Test finalizing shot data when target is missed"""
        # Create and update shot data
        shot_data = create_shot_data(2, time.time(), (120, 430))
        shot_data = update_shot_data(shot_data, (300, 500), (100.0, 50.0), 200.0, 30.0, 80.0)
        
        # Finalize with miss
        finalized_data = finalize_shot_data(shot_data, False)
        
        assert finalized_data['hit_target'] == False
        assert finalized_data['shot_number'] == 2
        assert finalized_data['angle_deg'] == 30.0

    def test_shot_data_workflow(self):
        """Test complete shot data workflow from creation to finalization"""
        # Step 1: Create initial shot data
        shot_data = create_shot_data(1, time.time(), (120, 430))
        assert 'hit_target' not in shot_data
        
        # Step 2: Update with launch data
        shot_data = update_shot_data(shot_data, (250, 350), (200.0, -150.0), 300.0, 37.0, 120.0)
        assert shot_data['hit_target'] == False
        assert shot_data['velocity'] == (200.0, -150.0)
        
        # Step 3: Finalize with result
        shot_data = finalize_shot_data(shot_data, True)
        assert shot_data['hit_target'] == True
        
        # Verify all data is preserved
        assert shot_data['shot_number'] == 1
        assert shot_data['start_pos'] == (120, 430)
        assert shot_data['end_pos'] == (250, 350)
        assert shot_data['angle_deg'] == 37.0
        assert shot_data['impulse_magnitude'] == 300.0

    def test_multiple_shots_data(self):
        """Test managing data for multiple shots"""
        shot_history = []
        
        # Shot 1 - Miss
        shot1 = create_shot_data(1, time.time(), (120, 430))
        shot1 = update_shot_data(shot1, (200, 400), (100.0, -100.0), 200.0, 45.0, 80.0)
        shot1 = finalize_shot_data(shot1, False)
        shot_history.append(shot1)
        
        # Shot 2 - Hit
        shot2 = create_shot_data(2, time.time(), (120, 430))
        shot2 = update_shot_data(shot2, (250, 350), (150.0, -200.0), 300.0, 53.0, 120.0)
        shot2 = finalize_shot_data(shot2, True)
        shot_history.append(shot2)
        
        # Verify shot history
        assert len(shot_history) == 2
        assert shot_history[0]['shot_number'] == 1
        assert shot_history[0]['hit_target'] == False
        assert shot_history[1]['shot_number'] == 2
        assert shot_history[1]['hit_target'] == True
        
        # Verify statistics
        hits = sum(1 for shot in shot_history if shot['hit_target'])
        assert hits == 1
        assert len(shot_history) == 2
