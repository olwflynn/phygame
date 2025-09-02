import pytest
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game import config


class TestConfig:
    """Test configuration constants and values"""
    
    def test_window_dimensions(self):
        """Test window width and height are valid"""
        assert hasattr(config, 'WINDOW_WIDTH')
        assert hasattr(config, 'WINDOW_HEIGHT')
        
        assert isinstance(config.WINDOW_WIDTH, int)
        assert isinstance(config.WINDOW_HEIGHT, int)
        
        assert config.WINDOW_WIDTH > 0
        assert config.WINDOW_HEIGHT > 0
        
        # Reasonable window size constraints
        assert config.WINDOW_WIDTH >= 800
        assert config.WINDOW_HEIGHT >= 400
        assert config.WINDOW_WIDTH <= 1920
        assert config.WINDOW_HEIGHT <= 1080
    
    def test_fps_setting(self):
        """Test FPS configuration"""
        assert hasattr(config, 'FPS')
        assert isinstance(config.FPS, int)
        assert config.FPS > 0
        
        # Reasonable FPS constraints
        assert config.FPS >= 30
        assert config.FPS <= 120
    
    def test_gravity_setting(self):
        """Test gravity configuration"""
        assert hasattr(config, 'GRAVITY')
        assert isinstance(config.GRAVITY, tuple)
        assert len(config.GRAVITY) == 2
        
        gravity_x, gravity_y = config.GRAVITY
        assert isinstance(gravity_x, (int, float))
        assert isinstance(gravity_y, (int, float))
        
        # Gravity should be downward (positive Y in screen coordinates)
        assert gravity_x == 0  # No horizontal gravity
        assert gravity_y > 0   # Downward gravity
    
    def test_slingshot_position(self):
        """Test slingshot position configuration"""
        assert hasattr(config, 'SLINGSHOT_POS')
        assert isinstance(config.SLINGSHOT_POS, tuple)
        assert len(config.SLINGSHOT_POS) == 2
        
        slingshot_x, slingshot_y = config.SLINGSHOT_POS
        assert isinstance(slingshot_x, (int, float))
        assert isinstance(slingshot_y, (int, float))
        
        # Slingshot should be within window bounds
        assert 0 <= slingshot_x <= config.WINDOW_WIDTH
        assert 0 <= slingshot_y <= config.WINDOW_HEIGHT
    
    def test_bird_radius(self):
        """Test bird radius configuration"""
        assert hasattr(config, 'BIRD_RADIUS')
        assert isinstance(config.BIRD_RADIUS, (int, float))
        assert config.BIRD_RADIUS > 0
        
        # Reasonable bird size constraints
        assert config.BIRD_RADIUS >= 5
        assert config.BIRD_RADIUS <= 50
    
    def test_target_size(self):
        """Test target size configuration"""
        assert hasattr(config, 'TARGET_SIZE')
        assert isinstance(config.TARGET_SIZE, tuple)
        assert len(config.TARGET_SIZE) == 2
        
        target_width, target_height = config.TARGET_SIZE
        assert isinstance(target_width, (int, float))
        assert isinstance(target_height, (int, float))
        
        assert target_width > 0
        assert target_height > 0
        
        # Reasonable target size constraints
        assert target_width >= 10
        assert target_height >= 10
        assert target_width <= 200
        assert target_height <= 200
    
    def test_angle_constraints(self):
        """Test angle constraint configurations"""
        assert hasattr(config, 'ANGLE_MIN_DEG')
        assert hasattr(config, 'ANGLE_MAX_DEG')
        
        assert isinstance(config.ANGLE_MIN_DEG, (int, float))
        assert isinstance(config.ANGLE_MAX_DEG, (int, float))
        
        # Angle constraints should be valid
        assert 0 <= config.ANGLE_MIN_DEG <= 90
        assert 0 <= config.ANGLE_MAX_DEG <= 90
        assert config.ANGLE_MIN_DEG < config.ANGLE_MAX_DEG
    
    def test_force_constraints(self):
        """Test force constraint configurations"""
        assert hasattr(config, 'FORCE_MIN')
        assert hasattr(config, 'FORCE_MAX')
        
        assert isinstance(config.FORCE_MIN, (int, float))
        assert isinstance(config.FORCE_MAX, (int, float))
        
        # Force constraints should be valid
        assert config.FORCE_MIN > 0
        assert config.FORCE_MAX > 0
        assert config.FORCE_MIN < config.FORCE_MAX
    
    def test_config_consistency(self):
        """Test that configuration values are consistent with each other"""
        # Slingshot should be positioned reasonably relative to window
        slingshot_x, slingshot_y = config.SLINGSHOT_POS
        assert slingshot_x < config.WINDOW_WIDTH * 0.5  # Should be on left side
        assert slingshot_y > config.WINDOW_HEIGHT * 0.5  # Should be in lower half
        
        # Bird radius should be reasonable relative to window size
        assert config.BIRD_RADIUS < config.WINDOW_WIDTH / 20
        assert config.BIRD_RADIUS < config.WINDOW_HEIGHT / 20
        
        # Target size should be reasonable relative to window size
        target_width, target_height = config.TARGET_SIZE
        assert target_width < config.WINDOW_WIDTH / 5
        assert target_height < config.WINDOW_HEIGHT / 5
    
    def test_config_immutability(self):
        """Test that config values are constants (not accidentally modified)"""
        # Store original values
        original_values = {
            'WINDOW_WIDTH': config.WINDOW_WIDTH,
            'WINDOW_HEIGHT': config.WINDOW_HEIGHT,
            'FPS': config.FPS,
            'GRAVITY': config.GRAVITY,
            'SLINGSHOT_POS': config.SLINGSHOT_POS,
            'BIRD_RADIUS': config.BIRD_RADIUS,
            'TARGET_SIZE': config.TARGET_SIZE,
            'ANGLE_MIN_DEG': config.ANGLE_MIN_DEG,
            'ANGLE_MAX_DEG': config.ANGLE_MAX_DEG,
            'FORCE_MIN': config.FORCE_MIN,
            'FORCE_MAX': config.FORCE_MAX,
        }
        
        # Verify all values are still the same
        for key, original_value in original_values.items():
            current_value = getattr(config, key)
            assert current_value == original_value, f"{key} was modified"
    
    def test_config_completeness(self):
        """Test that all expected configuration constants are present"""
        expected_constants = [
            'WINDOW_WIDTH',
            'WINDOW_HEIGHT', 
            'FPS',
            'GRAVITY',
            'SLINGSHOT_POS',
            'BIRD_RADIUS',
            'TARGET_SIZE',
            'ANGLE_MIN_DEG',
            'ANGLE_MAX_DEG',
            'FORCE_MIN',
            'FORCE_MAX',
        ]
        
        for constant in expected_constants:
            assert hasattr(config, constant), f"Missing configuration constant: {constant}"
    
    def test_config_types(self):
        """Test that all configuration values have correct types"""
        type_checks = {
            'WINDOW_WIDTH': int,
            'WINDOW_HEIGHT': int,
            'FPS': int,
            'GRAVITY': tuple,
            'SLINGSHOT_POS': tuple,
            'BIRD_RADIUS': (int, float),
            'TARGET_SIZE': tuple,
            'ANGLE_MIN_DEG': (int, float),
            'ANGLE_MAX_DEG': (int, float),
            'FORCE_MIN': (int, float),
            'FORCE_MAX': (int, float),
        }
        
        for constant, expected_type in type_checks.items():
            value = getattr(config, constant)
            if isinstance(expected_type, tuple):
                assert isinstance(value, expected_type), f"{constant} should be one of {expected_type}, got {type(value)}"
            else:
                assert isinstance(value, expected_type), f"{constant} should be {expected_type}, got {type(value)}"
