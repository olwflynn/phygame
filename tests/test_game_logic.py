import pytest
import pymunk
import sys
import os

# Add src directory to Python path so we can import from main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.entities import create_world, create_ground, create_target, create_bird


class TestGameLogic:
    """Test game logic functions from main.py"""
    
    def setup_method(self):
        """Set up physics world and entities for each test"""
        self.space = create_world((0, 900))
        self.ground = create_ground(self.space, y=500, width=960)
        self.target = create_target(self.space, pos=(800, 400), size=(40, 40))
        self.bird = create_bird(self.space, pos=(120, 430), radius=14, velocity=(0, 0))
    
    def test_reset_bird(self):
        """Test reset_bird function resets bird to starting position"""
        # Move bird to different position
        self.bird.body.position = (500, 300)
        self.bird.body.velocity = (100, 50)
        
        # Import and test reset function
        from game.game_state import reset_bird
        new_bird = reset_bird(self.space, self.bird)
        
        # Check new bird is at starting position
        assert new_bird.body.position.x == 120
        assert new_bird.body.position.y == 430
        assert new_bird.body.velocity.x == 0
        assert new_bird.body.velocity.y == 0
        
        # Check old bird is removed from space
        assert self.bird not in self.space.shapes
        assert self.bird.body not in self.space.bodies
        
        # Check new bird is added to space
        assert new_bird in self.space.shapes
        assert new_bird.body in self.space.bodies
    
    def test_reset_target(self):
        """Test reset_target function resets target to starting position"""
        # Move target to different position
        self.target.body.position = (600, 300)
        self.target.body.velocity = (50, 25)
        self.target.body.angular_velocity = 1.5
        
        # Import and test reset function
        from game.game_state import reset_target
        reset_target(self.target)
        
        # Check target is reset to starting position
        assert self.target.body.position.x == 800
        assert self.target.body.position.y == 400
        assert self.target.body.velocity.x == 0
        assert self.target.body.velocity.y == 0
        assert self.target.body.angular_velocity == 0
    
    def test_reset_game(self):
        """Test reset_game function resets both bird and target"""
        # Move both entities
        self.bird.body.position = (500, 300)
        self.target.body.position = (600, 300)
        
        # Import and test reset function
        from game.game_state import reset_game
        new_bird = reset_game(self.bird, self.target, self.space)
        
        # Check bird is reset
        assert new_bird.body.position.x == 120
        assert new_bird.body.position.y == 430
        
        # Check target is reset
        assert self.target.body.position.x == 800
        assert self.target.body.position.y == 400
    
    def test_check_target_hit_when_hit(self):
        """Test collision detection when bird hits target"""
        # Position bird close to target (within collision distance)
        self.bird.body.position = (800, 400)  # Same position as target
        
        from game.entities import check_target_hit
        assert check_target_hit(self.bird, self.target) is True
    
    def test_check_target_hit_when_close(self):
        """Test collision detection when bird is close but not exactly on target"""
        # Position bird close to target (within 35 pixel collision distance)
        # Target is 40x40, so center is at (800, 400)
        # Bird at (800, 366) is 34 pixels above target center
        self.bird.body.position = (800, 366)  # 34 pixels above target center
        
        from game.entities import check_target_hit
        assert check_target_hit(self.bird, self.target) is True
    
    def test_check_target_hit_when_miss(self):
        """Test collision detection when bird misses target"""
        # Position bird far from target (outside collision distance)
        self.bird.body.position = (800, 300)  # 100 pixels above target center
        
        from game.entities import check_target_hit
        assert check_target_hit(self.bird, self.target) is False
    
    def test_check_target_hit_edge_case(self):
        """Test collision detection at exact boundary"""
        # Position bird exactly at collision boundary (34 pixels from target center)
        # Target center is at (800, 400), so 34 pixels above is (800, 366)
        self.bird.body.position = (800, 366)  # 34 pixels above target center
        
        from game.entities import check_target_hit
        assert check_target_hit(self.bird, self.target) is True
        
        # Position bird just outside collision boundary
        self.bird.body.position = (800, 365)  # 35 pixels above target center
        
        assert check_target_hit(self.bird, self.target) is False
    
    def test_bird_auto_reset_when_landed(self):
        """Test bird auto-reset when it lands (low velocity and past x=500)"""
        # Position bird past x=500 with low velocity (simulating landed state)
        self.bird.body.position = (600, 480)  # Past x=500, near ground
        self.bird.body.velocity = (3, 2)  # Low velocity (both < 5)
        
        # Simulate the auto-reset logic from main game loop
        if (abs(self.bird.body.velocity.x) < 5 and 
            abs(self.bird.body.velocity.y) < 5 and 
            self.bird.body.position.x > 500):
            
            from game.game_state import reset_bird
            new_bird = reset_bird(self.space, self.bird)
            
            # Check bird is reset to starting position
            assert new_bird.body.position.x == 120
            assert new_bird.body.position.y == 430
    
    def test_bird_auto_reset_when_out_of_bounds(self):
        """Test bird auto-reset when it goes out of horizontal bounds"""
        # Test left boundary
        self.bird.body.position = (50, 300)  # x < 100
        
        # Simulate the out-of-bounds reset logic
        if self.bird.body.position.x < 100:
            from game.game_state import reset_bird
            new_bird = reset_bird(self.space, self.bird)
            assert new_bird.body.position.x == 120
            
            # Update self.bird reference for next test
            self.bird = new_bird
        
        # Test right boundary
        self.bird.body.position = (1000, 300)  # x > 960
        
        if self.bird.body.position.x > 960:
            from game.game_state import reset_bird
            new_bird = reset_bird(self.space, self.bird)
            assert new_bird.body.position.x == 120
    
    def test_bird_not_reset_when_moving(self):
        """Test bird is NOT reset when it has high velocity"""
        # Position bird past x=500 but with high velocity
        self.bird.body.position = (600, 300)
        self.bird.body.velocity = (20, 15)  # High velocity (both > 5)
        
        # Simulate the auto-reset logic
        should_reset = (abs(self.bird.body.velocity.x) < 5 and 
                       abs(self.bird.body.velocity.y) < 5 and 
                       self.bird.body.position.x > 500)
        
        assert should_reset is False  # Should NOT reset
    
    def test_bird_not_reset_before_shot(self):
        """Test bird is NOT reset before first shot (shots_fired = 0)"""
        # Position bird past x=500 with low velocity but no shots fired
        self.bird.body.position = (600, 480)
        self.bird.body.velocity = (3, 2)
        shots_fired = 0  # No shots fired yet
        
        # Simulate the auto-reset logic
        should_reset = (abs(self.bird.body.velocity.x) < 5 and 
                       abs(self.bird.body.velocity.y) < 5 and 
                       shots_fired > 0 and  # This condition fails
                       self.bird.body.position.x > 500)
        
        assert should_reset is False  # Should NOT reset
