import pytest
import sys
import os
import pygame
import matplotlib.pyplot as plt
from collections import deque
from unittest.mock import patch, MagicMock

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.ui import update_charts, create_shot_table, render_game
from game.ai import ShotSuggestion


class TestUpdateCharts:
    """Test chart update functionality"""
    
    def setup_method(self):
        """Set up test data for charts"""
        # Create mock figure and axes
        self.fig = MagicMock()
        self.ax1 = MagicMock()
        self.ax2 = MagicMock()
        self.ax3 = MagicMock()
        
        # Create test data
        self.time_data = deque([0.0, 0.1, 0.2, 0.3, 0.4], maxlen=300)
        self.x_pos_data = deque([120, 150, 180, 210, 240], maxlen=300)
        self.y_pos_data = deque([430, 420, 410, 400, 390], maxlen=300)
        self.x_vel_data = deque([300, 280, 260, 240, 220], maxlen=300)
        self.y_vel_data = deque([-200, -180, -160, -140, -120], maxlen=300)
    
    def test_update_charts_with_sufficient_data(self):
        """Test chart update with sufficient data"""
        # Should not raise any exceptions
        update_charts(
            self.fig, self.ax1, self.ax2, self.ax3,
            self.time_data, self.x_pos_data, self.y_pos_data,
            self.x_vel_data, self.y_vel_data
        )
        
        # Verify that clear was called on all axes
        self.ax1.clear.assert_called_once()
        self.ax2.clear.assert_called_once()
        self.ax3.clear.assert_called_once()
    
    def test_update_charts_with_insufficient_data(self):
        """Test chart update with insufficient data"""
        # Create data with only one point
        single_time = deque([0.0], maxlen=300)
        single_x = deque([120], maxlen=300)
        single_y = deque([430], maxlen=300)
        single_vx = deque([300], maxlen=300)
        single_vy = deque([-200], maxlen=300)
        
        # Should return early without updating charts
        update_charts(
            self.fig, self.ax1, self.ax2, self.ax3,
            single_time, single_x, single_y, single_vx, single_vy
        )
        
        # Axes should not be cleared with insufficient data
        self.ax1.clear.assert_not_called()
        self.ax2.clear.assert_not_called()
        self.ax3.clear.assert_not_called()
    
    def test_update_charts_with_empty_data(self):
        """Test chart update with empty data"""
        empty_data = deque(maxlen=300)
        
        # Should return early without updating charts
        update_charts(
            self.fig, self.ax1, self.ax2, self.ax3,
            empty_data, empty_data, empty_data, empty_data, empty_data
        )
        
        # Axes should not be cleared with empty data
        self.ax1.clear.assert_not_called()
        self.ax2.clear.assert_not_called()
        self.ax3.clear.assert_not_called()
    
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.draw')
    @patch('matplotlib.pyplot.pause')
    def test_update_charts_calls_matplotlib_functions(self, mock_pause, mock_draw, mock_tight_layout):
        """Test that update_charts calls matplotlib functions"""
        update_charts(
            self.fig, self.ax1, self.ax2, self.ax3,
            self.time_data, self.x_pos_data, self.y_pos_data,
            self.x_vel_data, self.y_vel_data
        )
        
        # Verify matplotlib functions are called
        mock_tight_layout.assert_called_once()
        mock_draw.assert_called_once()
        mock_pause.assert_called_once_with(0.001)


class TestCreateShotTable:
    """Test shot table creation functionality"""
    
    def test_create_shot_table_with_data(self):
        """Test creating shot table with shot history data"""
        shot_history = [
            {
                'shot_number': 1,
                'angle_deg': 45.0,
                'impulse_magnitude': 500.0,
                'hit_target': True
            },
            {
                'shot_number': 2,
                'angle_deg': 30.0,
                'impulse_magnitude': 400.0,
                'hit_target': False
            }
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = create_shot_table(shot_history)
            
            assert result == mock_fig
            mock_subplots.assert_called_once_with(figsize=(8, 6))
    
    def test_create_shot_table_with_empty_history(self):
        """Test creating shot table with empty history"""
        shot_history = []
        
        result = create_shot_table(shot_history)
        
        assert result is None
    
    def test_create_shot_table_with_none_history(self):
        """Test creating shot table with None history"""
        result = create_shot_table(None)
        
        assert result is None
    
    def test_create_shot_table_with_many_shots(self):
        """Test creating shot table with many shots (should limit to last 10)"""
        # Create 15 shots
        shot_history = []
        for i in range(15):
            shot_history.append({
                'shot_number': i + 1,
                'angle_deg': 45.0 + i,
                'impulse_magnitude': 500.0 + i * 10,
                'hit_target': i % 2 == 0
            })
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = create_shot_table(shot_history)
            
            assert result == mock_fig
            # Should only show last 10 shots
            mock_ax.table.assert_called_once()
    
    def test_create_shot_table_missing_fields(self):
        """Test creating shot table with missing fields in shot data"""
        shot_history = [
            {
                'shot_number': 1,
                # Missing angle_deg
                'impulse_magnitude': 500.0,
                'hit_target': True
            },
            {
                # Missing shot_number
                'angle_deg': 30.0,
                'impulse_magnitude': 400.0,
                'hit_target': False
            }
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = create_shot_table(shot_history)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()


class TestRenderGame:
    """Test game rendering functionality"""
    
    def setup_method(self):
        """Set up test data for rendering"""
        # Initialize pygame for testing
        pygame.init()
        
        # Create mock objects
        self.screen = MagicMock()
        self.space = MagicMock()
        self.bird = MagicMock()
        self.bird.body.position = MagicMock()
        self.bird.body.position.x = 120
        self.bird.body.position.y = 430
        self.target = MagicMock()
        self.target.body.position = MagicMock()
        self.target.body.position.x = 800
        self.target.body.position.y = 400
        self.obstacles = []
        self.font = MagicMock()
        self.suggestion_font = MagicMock()
        
        # Mock font rendering
        self.font.render.return_value = MagicMock()
        self.suggestion_font.render.return_value = MagicMock()
    
    def teardown_method(self):
        """Clean up after tests"""
        pygame.quit()
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_basic_parameters(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test basic game rendering with minimal parameters"""
        render_game(
            screen=self.screen,
            space=self.space,
            bird=self.bird,
            target=self.target,
            obstacles=self.obstacles,
            launching=False,
            start_pos=None,
            velocity_multiplier=5,
            score=0,
            shots_fired=0,
            max_shots=3,
            font=self.font,
            width=960,
            height=540,
            show_charts=False,
            show_table=False,
            show_suggestion=False,
            current_suggestion=None,
            suggestion_font=self.suggestion_font
        )
        
        # Verify screen.fill was called (background)
        self.screen.fill.assert_called_once()
        
        # Verify font.render was called for UI elements
        assert self.font.render.call_count >= 4  # Score, level, shots, settings
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_launching(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering when bird is being launched"""
        start_pos = (120, 430)
        
        with patch('pygame.mouse.get_pos', return_value=(200, 400)):
            render_game(
                screen=self.screen,
                space=self.space,
                bird=self.bird,
                target=self.target,
                obstacles=self.obstacles,
                launching=True,
                start_pos=start_pos,
                velocity_multiplier=5,
                score=0,
                shots_fired=0,
                max_shots=3,
                font=self.font,
                width=960,
                height=540,
                show_charts=False,
                show_table=False,
                show_suggestion=False,
                current_suggestion=None,
                suggestion_font=self.suggestion_font
            )
        
        # Should draw launch line when launching
        mock_line.assert_called()
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_obstacles(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering with obstacles"""
        # Create mock obstacle
        obstacle = MagicMock()
        obstacle.body.position = MagicMock()
        obstacle.body.position.x = 400
        obstacle.body.position.y = 450
        obstacles = [(obstacle, (20, 100))]
        
        render_game(
            screen=self.screen,
            space=self.space,
            bird=self.bird,
            target=self.target,
            obstacles=obstacles,
            launching=False,
            start_pos=None,
            velocity_multiplier=5,
            score=0,
            shots_fired=0,
            max_shots=3,
            font=self.font,
            width=960,
            height=540,
            show_charts=False,
            show_table=False,
            show_suggestion=False,
            current_suggestion=None,
            suggestion_font=self.suggestion_font
        )
        
        # Should draw obstacles
        mock_rect.assert_called()
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_suggestion(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering with AI suggestion displayed"""
        suggestion = ShotSuggestion(angle_deg=45.0, impulse_magnitude=600.0)
        
        render_game(
            screen=self.screen,
            space=self.space,
            bird=self.bird,
            target=self.target,
            obstacles=self.obstacles,
            launching=False,
            start_pos=None,
            velocity_multiplier=5,
            score=0,
            shots_fired=0,
            max_shots=3,
            font=self.font,
            width=960,
            height=540,
            show_charts=False,
            show_table=False,
            show_suggestion=True,
            current_suggestion=suggestion,
            suggestion_font=self.suggestion_font
        )
        
        # Should render suggestion text
        self.suggestion_font.render.assert_called()
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_episode_over(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering when episode is over"""
        render_game(
            screen=self.screen,
            space=self.space,
            bird=self.bird,
            target=self.target,
            obstacles=self.obstacles,
            launching=False,
            start_pos=None,
            velocity_multiplier=5,
            score=0,
            shots_fired=3,
            max_shots=3,
            font=self.font,
            width=960,
            height=540,
            show_charts=False,
            show_table=False,
            show_suggestion=False,
            current_suggestion=None,
            suggestion_font=self.suggestion_font,
            episode_over=True
        )
        
        # Should render game over message
        assert self.font.render.call_count >= 5  # Including game over message
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_different_scores(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering with different scores"""
        scores = [0, 100, 200, 500]
        
        for score in scores:
            self.font.reset_mock()
            
            render_game(
                screen=self.screen,
                space=self.space,
                bird=self.bird,
                target=self.target,
                obstacles=self.obstacles,
                launching=False,
                start_pos=None,
                velocity_multiplier=5,
                score=score,
                shots_fired=0,
                max_shots=3,
                font=self.font,
                width=960,
                height=540,
                show_charts=False,
                show_table=False,
                show_suggestion=False,
                current_suggestion=None,
                suggestion_font=self.suggestion_font
            )
            
            # Should render score
            assert self.font.render.call_count >= 4
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_different_shots_fired(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering with different shots fired counts"""
        shots_fired_values = [0, 1, 2, 3]
        
        for shots_fired in shots_fired_values:
            self.font.reset_mock()
            
            render_game(
                screen=self.screen,
                space=self.space,
                bird=self.bird,
                target=self.target,
                obstacles=self.obstacles,
                launching=False,
                start_pos=None,
                velocity_multiplier=5,
                score=0,
                shots_fired=shots_fired,
                max_shots=3,
                font=self.font,
                width=960,
                height=540,
                show_charts=False,
                show_table=False,
                show_suggestion=False,
                current_suggestion=None,
                suggestion_font=self.suggestion_font
            )
            
            # Should render shots remaining
            assert self.font.render.call_count >= 4
    
    @patch('pygame.draw.circle')
    @patch('pygame.draw.rect')
    @patch('pygame.draw.polygon')
    @patch('pygame.draw.line')
    def test_render_game_with_different_level_info(self, mock_line, mock_polygon, mock_rect, mock_circle):
        """Test game rendering with different level information"""
        level_numbers = [1, 5, 10]
        level_types = ["LLM", "Predefined"]
        
        for level_number in level_numbers:
            for level_type in level_types:
                self.font.reset_mock()
                
                render_game(
                    screen=self.screen,
                    space=self.space,
                    bird=self.bird,
                    target=self.target,
                    obstacles=self.obstacles,
                    launching=False,
                    start_pos=None,
                    velocity_multiplier=5,
                    score=0,
                    shots_fired=0,
                    max_shots=3,
                    font=self.font,
                    width=960,
                    height=540,
                    show_charts=False,
                    show_table=False,
                    show_suggestion=False,
                    current_suggestion=None,
                    suggestion_font=self.suggestion_font,
                    level_number=level_number,
                    level_type=level_type
                )
                
                # Should render level information
                assert self.font.render.call_count >= 4


class TestUIIntegration:
    """Integration tests for UI functionality"""
    
    def setup_method(self):
        """Set up for integration tests"""
        pygame.init()
    
    def teardown_method(self):
        """Clean up after integration tests"""
        pygame.quit()
    
    def test_ui_functions_import_correctly(self):
        """Test that all UI functions can be imported"""
        from game.ui import update_charts, create_shot_table, render_game
        
        assert callable(update_charts)
        assert callable(create_shot_table)
        assert callable(render_game)
    
    def test_ui_functions_handle_edge_cases(self):
        """Test UI functions handle edge cases gracefully"""
        # Test with None values
        assert create_shot_table(None) is None
        assert create_shot_table([]) is None
        
        # Test with invalid data types
        with pytest.raises((TypeError, AttributeError)):
            create_shot_table("invalid_data")
    
    @patch('matplotlib.pyplot.subplots')
    def test_shot_table_with_realistic_data(self, mock_subplots):
        """Test shot table with realistic shot data"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Realistic shot history
        shot_history = [
            {
                'shot_number': 1,
                'angle_deg': 45.2,
                'impulse_magnitude': 523.7,
                'hit_target': True
            },
            {
                'shot_number': 2,
                'angle_deg': 38.9,
                'impulse_magnitude': 487.3,
                'hit_target': False
            },
            {
                'shot_number': 3,
                'angle_deg': 52.1,
                'impulse_magnitude': 612.8,
                'hit_target': True
            }
        ]
        
        result = create_shot_table(shot_history)
        
        assert result == mock_fig
        mock_subplots.assert_called_once()
        mock_ax.table.assert_called_once()
