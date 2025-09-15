import json
import pymunk
import openai
import os
import random
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional, Union

from .game_state import reset_bird, reset_target

# Load environment variables from .env file
load_dotenv()

def generate_level_with_llm() -> Dict[str, Any]:
    """Generate a new level using OpenAI's GPT model"""
    try:
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = """
        Generate a new level for an Angry Birds-style physics game. The level should be challenging but fair.
        
        Requirements:
        - The game area is 960x540 pixels
        - Bird starts at position (120, 430)
        - Ground is at y=500
        - Target should be a house that can be hit
        - Obstacles should be rectangular blocks that can be destroyed or used as cover
        - Minimum 4 and Maximum 10 obstacles
        - Target should be reachable with the given physics constraints
        - There should be at least one obstacle on the ground (y=500) to make it challenging
        
        Return ONLY a valid JSON object with this exact structure:
        {
          "targets": [
            {"x": <x_position>, "y": <y_position>}
          ],
          "obstacles": [
            {"x": <center_x>, "y": <center_y>, "w": <width>, "h": <height>}
          ]
        }
        
        Constraints:
        - Target x: 400-900, y: 300-480
        - Obstacle x: 200-800, y: 300-480
        - Obstacle dimensions: 10-100 pixels
        - Ensure obstacles don't completely block the target
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a game level designer. Generate only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        print(response.choices[0].message.content)
        # Extract JSON from response
        level_json = response.choices[0].message.content.strip()
        
        # Parse and validate the JSON
        level_data = json.loads(level_json)
        
        # Validate structure
        if "targets" not in level_data or "obstacles" not in level_data:
            raise ValueError("Invalid level structure")
        
        return level_data
        
    except Exception as e:
        print(f"Error generating level with LLM: {e}")
        # Fallback to a default level
        return get_default_level()

def get_default_level() -> Dict[str, Any]:
    """Return a default level configuration"""
    return {
        "targets": [{"x": 800, "y": 400}],
        "obstacles": [
            {"x": 400, "y": 500, "w": 10, "h": 100}
        ]
    }

def parse_level_from_json(json_string: str) -> Dict[str, Any]:
    """Parse level data from a JSON string"""
    try:
        level_data = json.loads(json_string)
        
        # Validate structure
        if "targets" not in level_data or "obstacles" not in level_data:
            raise ValueError("Invalid level structure: missing 'targets' or 'obstacles'")
        
        # Validate targets
        for target in level_data["targets"]:
            if "x" not in target or "y" not in target:
                raise ValueError("Invalid target structure: missing 'x' or 'y'")
        
        # Validate obstacles
        for obstacle in level_data["obstacles"]:
            if "x" not in obstacle or "y" not in obstacle or "w" not in obstacle or "h" not in obstacle:
                raise ValueError("Invalid obstacle structure: missing required fields")
        
        return level_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing level data: {e}")

def load_level(space, level_data: Optional[Union[Dict[str, Any], str]] = None, use_llm: bool = True, prev_bird: pymunk.Circle = None, prev_target: pymunk.Poly = None):
    """
    Parse JSON and spawn objects into pymunk space
    
    Args:
        space: Pymunk space to add objects to
        level_data: Either a dictionary, JSON string, or None
        use_llm: If True and level_data is None, use LLM to generate level
    """
    if level_data is None:
        if use_llm:
            level_data = generate_level_with_llm()
        else:
            level_data = get_default_level()
    elif isinstance(level_data, str):
        # Parse JSON string
        level_data = parse_level_from_json(level_data)
    elif not isinstance(level_data, dict):
        raise ValueError("level_data must be a dictionary, JSON string, or None")

    # Clear existing obstacles, targets, and birds (keep ground)
    # Only remove obstacles (STATIC bodies that are not the static_body)
    for body in space.bodies[:]:  # Copy list to avoid modification during iteration
        if body != space.static_body and body.body_type == pymunk.Body.STATIC:
            for shape in body.shapes:
                space.remove(shape)
            space.remove(body)

    bird = reset_bird(space, prev_bird, (120, 430))

    # --- Targets ---
    targets = []
    for t in level_data.get("targets", []):
        target = reset_target(space, prev_target, (t["x"], t["y"]))
        targets.append(target)

    # --- Obstacles ---
    obstacles = []
    for o in level_data.get("obstacles", []):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (o["x"], o["y"])
        shape = pymunk.Poly.create_box(body, size=(o["w"], o["h"]))
        space.add(body, shape)
        obstacles.append((shape, (o["w"], o["h"])))

    return bird, targets, obstacles

def load_next_level(space, use_llm: bool = True, prev_bird: pymunk.Circle = None, prev_target: pymunk.Poly = None):
    """
    Generate and load a new level
    
    Args:
        space: Pymunk space to add objects to
        use_llm: If True, use LLM to generate level; if False, use default level
    """
    if use_llm:
        level_data = generate_level_with_llm()
    else:
        level_data = generate_random_level()
    
    return load_level(space, level_data, prev_bird=prev_bird, prev_target=prev_target)


def generate_random_level() -> Dict[str, Any]:
    """Generate a random level for the physics game"""

    GROUND_Y = 500
    
    # Target constraints (house should be reachable)
    TARGET_MIN_X = 400
    TARGET_MAX_X = 900
    TARGET_MIN_Y = 300
    TARGET_MAX_Y = 480
    
    # Obstacle constraints
    OBSTACLE_MIN_X = 200
    OBSTACLE_MAX_X = 800
    OBSTACLE_MIN_Y = 300
    OBSTACLE_MAX_Y = 480
    OBSTACLE_MIN_SIZE = 10
    OBSTACLE_MAX_SIZE = 100
    
    # Generate target (house)
    target_x = random.randint(TARGET_MIN_X, TARGET_MAX_X)
    target_y = random.randint(TARGET_MIN_Y, TARGET_MAX_Y)
    
    # Generate obstacles (4-10 obstacles)
    num_obstacles = random.randint(4, 10)
    obstacles = []
    
    # Ensure at least one obstacle is on the ground for challenge
    ground_obstacle = {
        "x": random.randint(OBSTACLE_MIN_X, OBSTACLE_MAX_X),
        "y": GROUND_Y,
        "w": random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE),
        "h": random.randint(OBSTACLE_MIN_SIZE, min(OBSTACLE_MAX_SIZE, GROUND_Y - OBSTACLE_MIN_Y))
    }
    obstacles.append(ground_obstacle)
    
    # Generate remaining obstacles
    for _ in range(num_obstacles - 1):
        obstacle = {
            "x": random.randint(OBSTACLE_MIN_X, OBSTACLE_MAX_X),
            "y": random.randint(OBSTACLE_MIN_Y, OBSTACLE_MAX_Y),
            "w": random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE),
            "h": random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        }
        obstacles.append(obstacle)
    
    return {
        "targets": [{"x": target_x, "y": target_y}],
        "obstacles": obstacles
    }


