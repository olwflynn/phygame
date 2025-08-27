import os
import sys

import pygame


def main() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    try:
        width, height = 960, 540
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PhyGame – Angry Birds Style (Skeleton)")
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((240, 240, 240))

            # Temporary ground line
            pygame.draw.line(screen, (50, 50, 50), (0, height - 50), (width, height - 50), 3)

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
