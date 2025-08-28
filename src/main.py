import os
import sys

import pygame
from game.physics import create_world
from game.entities import create_ground, create_target, create_bird
import pymunk
import pymunk.pygame_util

def main() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    try:
        width, height = 960, 540
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PhyGame – Angry Birds Style")
        clock = pygame.time.Clock()
        
        # Initialize font for score display
        font = pygame.font.Font(None, 36)
        
        # Create physics world and entities
        space = create_world((0, 900))  # Gravity: 900 pixels/s² downward
        ground = create_ground(space, y=500, width=960)  # Ground at y=500, spans full width
        target = create_target(space, pos=(800, 400), size=(40, 40))  # Target at (800, 400)
        bird = create_bird(space, pos=(120, 430), radius=14, velocity=(0,0))  # Bird at (120, 430)

        # Game state
        running = True
        launching = False
        start_pos = None
        score = 0
        shots_fired = 0
        max_shots = 3
        birds = []
        birds.append(bird)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if shots_fired < max_shots and bird.body.position.x < 500:
                        start_pos = pygame.mouse.get_pos()
                        launching = True

                elif event.type == pygame.MOUSEBUTTONUP and launching:
                    end_pos = pygame.mouse.get_pos()

                    # Calculate launch direction: from start to end (opposite of current)
                    dx = start_pos[0] - end_pos[0]  # Reverse X direction (left drag = right launch)
                    dy = start_pos[1] - end_pos[1]  # Reverse Y direction (up drag = down launch)
                    velocity = (dx * 3, dy * 3)
                    bird.body.apply_impulse_at_local_point(velocity)
                    
                    shots_fired += 1

                    # Calculate force magnitude based on drag distance
                    force_magnitude = min(((dx**2 + dy**2)**0.5) * 0.5, 800)
                    print(f"Force magnitude: {force_magnitude}")
                    print(f"dx: {dx}, dy: {dy}")
                    print(f"Drag: start={start_pos} -> end={end_pos}")
                    print(f"Launch: bird will go ({dx}, {dy}) direction")
                    
                    launching = False

                    print(f"\n=== SPACE CONTENTS ===")
                    print(f"Bodies: {len(space.bodies)}, Shapes: {len(space.shapes)}")
                    for i, body in enumerate(space.bodies):
                        body_type = "DYNAMIC" if body.body_type == body.DYNAMIC else "STATIC"
                        print(f"  Body {i}: pos=({body.position.x:.1f}, {body.position.y:.1f}), "
                                f"type={body_type}, mass={body.mass}")
                    print("=====================\n")

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset button
                        bird = reset_game(bird, target, space)
                        score = 0
                        shots_fired = 0

            # Step physics simulation (60 FPS)
            space.step(1/60)
            
            # Check for target hit
            if check_target_hit(bird, target):
                score += 100
                reset_target(target)
                bird = reset_bird(space, bird)
                shots_fired = 0  # Reset shots for new target

            # Reset bird if it's moving very slowly (landed)
            if (abs(bird.body.velocity.x) < 5 and 
                abs(bird.body.velocity.y) < 5 and 
                shots_fired > 0 and
                bird.body.position.x > 500):
                
                bird = reset_bird(space, bird)
                print("Bird auto-reset - landed")

            # Render everything
            render_game(screen, space, bird, target, launching, start_pos, score, shots_fired, max_shots, font, width, height)

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


def reset_bird(space, bird):
    space.remove(bird.body)
    space.remove(bird)
    bird = create_bird(space, pos=(120, 430), radius=14, velocity= (0,0))  # Bird at (120, 430)
    return bird


def reset_target(target):
    """Reset target to starting position"""
    target.body.position = (800, 400)
    target.body.velocity = (0, 0)
    target.body.angular_velocity = 0


def reset_game(bird, target, space):
    """Reset entire game state"""
    bird = reset_bird(space, bird)
    reset_target(target)
    return bird


def check_target_hit(bird, target):
    """Check if bird hit the target"""
    bird_pos = bird.body.position
    target_pos = target.body.position
    
    # Simple distance-based collision detection
    distance = ((bird_pos.x - target_pos.x)**2 + (bird_pos.y - target_pos.y)**2)**0.5
    return distance < 30  # Bird radius + half target size


def render_game(screen, space, bird, target, launching, start_pos, score, shots_fired, max_shots, font, width, height):
    """Render the entire game"""
    # Draw sky background (gradient effect)
    for y in range(540):
        # Sky gets darker towards the bottom
        blue_intensity = max(100, 235 - int(y * 0.2))
        pygame.draw.line(screen, (100, 150, blue_intensity), (0, y), (960, y))
    
    # Draw clouds (simple circles)
    pygame.draw.circle(screen, (255, 255, 255), (200, 100), 30)
    pygame.draw.circle(screen, (255, 255, 255), (230, 100), 25)
    pygame.draw.circle(screen, (255, 255, 255), (260, 100), 20)
    
    pygame.draw.circle(screen, (255, 255, 255), (700, 80), 25)
    pygame.draw.circle(screen, (255, 255, 255), (730, 80), 20)
    
    # Draw trees in background
    for x in [100, 300, 500, 700, 900]:
        # Tree trunk
        pygame.draw.rect(screen, (101, 67, 33), (x-10, 400, 20, 100))
        # Tree foliage
        pygame.draw.circle(screen, (34, 139, 34), (x, 380), 40)
        pygame.draw.circle(screen, (0, 100, 0), (x, 350), 30)
    
    # Draw ground with grass
    pygame.draw.rect(screen, (34, 139, 34), (0, 500, 960, 40))  # Grass
    pygame.draw.rect(screen, (139, 69, 19), (0, 540, 960, 20))  # Dirt
    
    # Draw target (house)
    target_pos = target.body.position
    # House base
    pygame.draw.rect(screen, (255, 218, 185), (int(target_pos.x-20), int(target_pos.y-20), 40, 40))
    # House roof
    pygame.draw.polygon(screen, (139, 69, 19), [
        (int(target_pos.x-25), int(target_pos.y-20)),
        (int(target_pos.x+25), int(target_pos.y-20)),
        (int(target_pos.x), int(target_pos.y-40))
    ])
    # House door
    pygame.draw.rect(screen, (101, 67, 33), (int(target_pos.x-8), int(target_pos.y-5), 16, 25))
    
    # Draw bird (simple colored circle for now - replace with image later)
    bird_pos = bird.body.position
    pygame.draw.circle(screen, (255, 165, 0), (int(bird_pos.x), int(bird_pos.y)), 14)  # Orange bird
    pygame.draw.circle(screen, (255, 69, 0), (int(bird_pos.x), int(bird_pos.y)), 14, 2)  # Red border
    
    # Draw launch line while dragging
    if launching and start_pos:
        current_pos = pygame.mouse.get_pos()
        # Draw dashed line to show launch direction
        pygame.draw.line(screen, (255, 255, 0), start_pos, current_pos, 3)
        # Draw arrowhead to show direction
        if current_pos != start_pos:
            line_dx = current_pos[0] - start_pos[0]
            line_dy = current_pos[1] - start_pos[1]
            # Normalize and scale for arrowhead
            length = (line_dx**2 + line_dy**2)**0.5
            if length > 0:
                dx_norm = line_dx / length * 20
                dy_norm = line_dy / length * 20
                arrow_tip = (current_pos[0] + dx_norm, current_pos[1] + dy_norm)
                pygame.draw.line(screen, (255, 255, 0), current_pos, arrow_tip, 3)
    
    # Draw UI elements
    # Score display
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (20, 20))
    
    # Shots remaining
    shots_text = font.render(f"Shots: {max_shots - shots_fired}/{max_shots}", True, (255, 255, 255))
    screen.blit(shots_text, (20, 60))
    
    # Reset button
    reset_text = font.render("Press R to Reset", True, (255, 0, 0))
    screen.blit(reset_text, (20, 100))
    
    # Game over message
    if shots_fired >= max_shots:
        game_over_text = font.render("Game Over! Press R to restart", True, (255, 0, 0))
        screen.blit(game_over_text, (width//2 - 150, height//2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
