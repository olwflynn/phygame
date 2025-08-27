import os
import sys

import pygame
from game.physics import create_world
from game.entities import create_ground, create_target, create_bird


def main() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    try:
        width, height = 960, 540
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PhyGame – Angry Birds Style (Skeleton)")
        clock = pygame.time.Clock()

        # Create physics world and entities
        space = create_world((0, 900))  # Gravity: 900 pixels/s² downward
        ground = create_ground(space, y=500, width=960)  # Ground at y=500, spans full width
        target = create_target(space, pos=(800, 400), size=(40, 40))  # Target at (800, 400)
        bird = create_bird(space, pos=(120, 430), radius=14)  # Bird at (120, 430)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Step physics simulation (60 FPS)
            space.step(1/60)
            
            # Clear screen with light gray background
            screen.fill((240, 240, 240))

            # Render ground (segment) - extract start and end points from pymunk segment
            ground_start = ground.a  # First endpoint: (0, 500)
            ground_end = ground.b    # Second endpoint: (960, 500)
            pygame.draw.line(screen, (139, 69, 19), ground_start, ground_end, 10)  # Brown ground, 10px thick
            
            # Render target (dynamic box) - get current physics position
            target_pos = target.body.position  # Current physics position (x, y)
            target_size = (40, 40)  # Same size as created
            
            # Create pygame rect centered on physics position
            target_rect = pygame.Rect(
                int(target_pos.x - target_size[0]/2),  # Center horizontally
                int(target_pos.y - target_size[1]/2),  # Center vertically
                target_size[0], target_size[1]
            )
            
            # Draw target with red fill and dark red border
            pygame.draw.rect(screen, (255, 0, 0), target_rect)  # Red fill
            pygame.draw.rect(screen, (139, 0, 0), target_rect, 3)  # Dark red border

            # Render bird (dynamic circle) - get current physics position
            bird_pos = bird.body.position  # Current physics position (x, y)
            bird_radius = bird.radius  # Same radius as created
            
            # Create pygame circle centered on physics position
            bird_circle = pygame.draw.circle(
                screen, (0, 255, 0), (int(bird_pos.x), int(bird_pos.y)), bird_radius)  # Green circle, 1px thick
            
            # # Draw bird with green fill and dark green border
            # pygame.draw.circle(screen, (0, 255, 0), bird_circle, bird_radius)  # Green fill
            # pygame.draw.circle(screen, (0, 139, 0), bird_circle, bird_radius, 1)  # Dark green border

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
