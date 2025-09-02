import json
import pymunk
import openai
import os
from typing import List, Tuple, Dict, Any, Optional, Union

from .game_state import reset_bird, reset_target

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
        - Maximum 5 obstacles to keep it manageable
        - Target should be reachable with the given physics constraints
        
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
        level_data = get_default_level()
    
    return load_level(space, level_data, prev_bird=prev_bird, prev_target=prev_target)

def load_custom_level(space, json_string: str):
    """
    Load a custom level from JSON string
    
    Args:
        space: Pymunk space to add objects to
        json_string: JSON string containing level data
    """
    level_data = parse_level_from_json(json_string)
    return load_level(space, level_data)


# --- Example usage and predefined levels ---
DEFAULT_LEVEL_JSON = """
{
  "targets": [
    {"x": 800, "y": 400}
  ],
  "obstacles": [
    {"x": 400, "y": 500, "w": 10, "h": 100}
  ]
}
"""

EASY_LEVEL_JSON = """
{
  "targets": [
    {"x": 600, "y": 450}
  ],
  "obstacles": [
    {"x": 400, "y": 500, "w": 10, "h": 50}
  ]
}
"""

HARD_LEVEL_JSON = """
{
  "targets": [
    {"x": 850, "y": 350}
  ],
  "obstacles": [
    {"x": 400, "y": 500, "w": 10, "h": 100},
    {"x": 500, "y": 450, "w": 10, "h": 80},
    {"x": 600, "y": 400, "w": 10, "h": 60},
    {"x": 700, "y": 350, "w": 10, "h": 40},
    {"x": 800, "y": 500, "w": 10, "h": 100}
  ]
}
"""

# Predefined level configurations
PREDEFINED_LEVELS = {
    "default": DEFAULT_LEVEL_JSON,
    "easy": EASY_LEVEL_JSON,
    "hard": HARD_LEVEL_JSON
}

def load_predefined_level(space, level_name: str):
    """
    Load a predefined level by name
    
    Args:
        space: Pymunk space to add objects to
        level_name: Name of predefined level ("default", "easy", "hard")
    """
    if level_name not in PREDEFINED_LEVELS:
        raise ValueError(f"Unknown level name: {level_name}. Available: {list(PREDEFINED_LEVELS.keys())}")
    
    json_string = PREDEFINED_LEVELS[level_name]
    return load_custom_level(space, json_string)

