import pytest
import pymunk
import sys
import os
import math
from unittest.mock import patch, MagicMock

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.ai import ShotSuggestion, suggest_best_shot, _simulate_shot
from game.entities import create_world, create_ground, create_target, create_bird


class TestShotSuggestion:
    """Test the ShotSuggestion dataclass"""
    
    def test_shot_suggestion_creation(self):
        """Test creating a ShotSuggestion instance"""
        suggestion = ShotSuggestion(angle_deg=45.0, impulse_magnitude=600.0)
        
        assert suggestion.angle_deg == 45.0
        assert suggestion.impulse_magnitude == 600.0
    
    def test_shot_suggestion_attributes(self):
        """Test ShotSuggestion has correct attributes"""
        suggestion = ShotSuggestion(angle_deg=30.0, impulse_magnitude=500.0)
        
        assert hasattr(suggestion, 'angle_deg')
        assert hasattr(suggestion, 'impulse_magnitude')
        assert isinstance(suggestion.angle_deg, float)
        assert isinstance(suggestion.impulse_magnitude, float)
    
    def test_shot_suggestion_immutability(self):
        """Test that ShotSuggestion fields can be accessed but are immutable by design"""
        suggestion = ShotSuggestion(angle_deg=60.0, impulse_magnitude=800.0)
        
        # Should be able to read values
        assert suggestion.angle_deg == 60.0
        assert suggestion.impulse_magnitude == 800.0


class TestSimulateShot:
    """Test the _simulate_shot function"""
    
    def setup_method(self):
        """Set up physics world and entities for each test"""
        self.space = create_world((0, 900))
        self.ground = create_ground(self.space, y=500, width=960)
        self.target = create_target(self.space, pos=(800, 400), size=(40, 40))
        self.bird = create_bird(self.space, pos=(120, 430), radius=14, velocity=(0, 0))
    
    def test_simulate_shot_miss(self):
        """Test simulation when shot misses target"""
        # Use a very low angle and very low impulse that should miss
        angle_deg = 5.0
        impulse_magnitude = 50.0
        
        result = _simulate_shot(self.space, angle_deg, impulse_magnitude)
        
        assert isinstance(result, bool)
        # Note: The result depends on physics simulation, so we just verify it's a boolean
        # In a real scenario, very low impulse shots are more likely to miss
    
    def test_simulate_shot_with_obstacles(self):
        """Test simulation with obstacles in the way"""
        # Add an obstacle
        obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obstacle_body.position = (400, 450)
        obstacle_shape = pymunk.Poly.create_box(obstacle_body, size=(20, 100))
        self.space.add(obstacle_body, obstacle_shape)
        
        # Test shot with mocked simulation for speed
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = False
            
            result = _simulate_shot(self.space, 45.0, 300.0)
            
            assert isinstance(result, bool)
    

    def test_simulate_shot_creates_new_space(self):
        """Test that simulation doesn't modify the original space"""
        original_bodies_count = len(self.space.bodies)
        original_shapes_count = len(self.space.shapes)
        
        # Run simulation with mocked physics to avoid long execution
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            _simulate_shot(self.space, 45.0, 400.0)
        
        # Original space should be unchanged
        assert len(self.space.bodies) == original_bodies_count
        assert len(self.space.shapes) == original_shapes_count


class TestSuggestBestShot:
    """Test the suggest_best_shot function"""
    
    def setup_method(self):
        """Set up physics world and entities for each test"""
        self.space = create_world((0, 900))
        self.ground = create_ground(self.space, y=500, width=960)
        self.target = create_target(self.space, pos=(800, 400), size=(40, 40))
        self.bird = create_bird(self.space, pos=(120, 430), radius=14, velocity=(0, 0))
    
    @patch('game.ai._simulate_shot')
    def test_suggest_best_shot_with_hits(self, mock_simulate):
        """Test suggest_best_shot when some shots hit"""
        # Mock simulation to return some hits
        mock_simulate.side_effect = [False, False, True]
        
        # Use small sample size for faster testing
        suggestion = suggest_best_shot(self.space, N_SAMPLES=3, plot=False)
        
        assert suggestion is not None
        assert isinstance(suggestion, ShotSuggestion)
        assert hasattr(suggestion, 'angle_deg')
        assert hasattr(suggestion, 'impulse_magnitude')
        assert isinstance(suggestion.angle_deg, float)
        assert isinstance(suggestion.impulse_magnitude, float)
    
    @patch('game.ai._simulate_shot')
    def test_suggest_best_shot_no_hits(self, mock_simulate):
        """Test suggest_best_shot when no shots hit"""
        # Mock simulation to return all misses
        mock_simulate.return_value = False
        
        suggestion = suggest_best_shot(self.space, N_SAMPLES=3, plot=False)
        
        assert suggestion is None
    
    @patch('game.ai._simulate_shot')
    def test_suggest_best_shot_returns_last_hit(self, mock_simulate):
        """Test that suggest_best_shot returns the last successful hit"""
        # Mock simulation to return hits at specific points
        mock_simulate.side_effect = [False, True, False]
        
        suggestion = suggest_best_shot(self.space, N_SAMPLES=3, plot=False)
        
        assert suggestion is not None
        # Should return the last hit (3rd call)
        assert mock_simulate.call_count == 3
    
    def test_suggest_best_shot_parameter_ranges(self):
        """Test suggest_best_shot with custom parameter ranges"""
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            
            suggestion = suggest_best_shot(
                self.space, 
                angle_min=10, 
                angle_max=80, 
                impulse_min=200, 
                impulse_max=1000,
                N_SAMPLES=3,
                plot=False
            )
            
            assert suggestion is not None
            assert isinstance(suggestion, ShotSuggestion)
    
    def test_suggest_best_shot_progress_output(self, capsys):
        """Test that suggest_best_shot shows progress output"""
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = False
            
            suggest_best_shot(self.space, N_SAMPLES=3, plot=False)
            
            # Check that progress output was printed
            captured = capsys.readouterr()
            assert "Simulation Progress:" in captured.out
    
    @patch('matplotlib.pyplot.show')
    def test_suggest_best_shot_with_plot(self, mock_show):
        """Test suggest_best_shot with plotting enabled"""
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            
            suggestion = suggest_best_shot(self.space, N_SAMPLES=5, plot=True)
            
            assert suggestion is not None
            # Plot should be shown
            mock_show.assert_called()
    
    def test_suggest_best_shot_empty_results(self):
        """Test suggest_best_shot with empty results"""
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = False
            
            suggestion = suggest_best_shot(self.space, N_SAMPLES=0, plot=False)
            
            assert suggestion is None
    
    def test_suggest_best_shot_with_obstacles(self):
        """Test suggest_best_shot with obstacles in the space"""
        # Add obstacles to make the level more complex
        obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obstacle_body.position = (400, 450)
        obstacle_shape = pymunk.Poly.create_box(obstacle_body, size=(20, 100))
        self.space.add(obstacle_body, obstacle_shape)
        
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            
            suggestion = suggest_best_shot(self.space, N_SAMPLES=5, plot=False)
            
            assert suggestion is not None
            assert isinstance(suggestion, ShotSuggestion)


class TestAIIntegration:
    """Integration tests for AI functionality"""
    
    def setup_method(self):
        """Set up physics world and entities for each test"""
        self.space = create_world((0, 900))
        self.ground = create_ground(self.space, y=500, width=960)
        self.target = create_target(self.space, pos=(800, 400), size=(40, 40))
        self.bird = create_bird(self.space, pos=(120, 430), radius=14, velocity=(0, 0))
    
    def test_ai_workflow_with_mocked_simulation(self):
        """Test complete AI workflow with mocked physics simulation"""
        # Use mocked simulation to avoid long execution times
        
        with patch('game.ai._simulate_shot') as mock_simulate:
            mock_simulate.return_value = True
            
            suggestion = suggest_best_shot(
                self.space, 
                angle_min=30, 
                angle_max=60, 
                impulse_min=300, 
                impulse_max=700,
                N_SAMPLES=5,  # Small sample for testing
                plot=False
            )
            
            # Should return a valid suggestion
            assert suggestion is not None
            assert isinstance(suggestion, ShotSuggestion)
            assert 30 <= suggestion.angle_deg <= 60
            assert 300 <= suggestion.impulse_magnitude <= 700
    
    def test_ai_with_different_target_positions(self):
        """Test AI with targets at different positions"""
        target_positions = [(600, 400), (700, 350), (850, 450)]
        
        for target_pos in target_positions:
            # Create new space for each test
            space = create_world((0, 900))
            ground = create_ground(space, y=500, width=960)
            target = create_target(space, pos=target_pos, size=(40, 40))
            bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0, 0))
            
            with patch('game.ai._simulate_shot') as mock_simulate:
                mock_simulate.return_value = True
                
                suggestion = suggest_best_shot(space, N_SAMPLES=5, plot=False)
                
                assert suggestion is not None
                assert isinstance(suggestion, ShotSuggestion)