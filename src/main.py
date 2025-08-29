import os
import sys

import pygame
import pymunk
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from game.physics import create_world
from game.entities import create_ground, create_target, create_bird
from game.ai import suggest_best_shot


def update_charts(fig, ax1, ax2, ax3, time_data, x_pos_data, y_pos_data, x_vel_data, y_vel_data):
    """Update and render the charts in a separate window"""
    if len(time_data) < 2:
        return
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Convert to relative time (seconds from start)
    start_time = time_data[0]
    relative_times = [t - start_time for t in time_data]
    
    # Position chart
    ax1.set_title('Bird Position Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Position (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 600)
    ax1.plot(relative_times, x_pos_data, 'b-', label='X Position', linewidth=2)
    ax1.plot(relative_times, y_pos_data, 'r-', label='Y Position', linewidth=2)
    ax1.legend()
    
    # Velocity chart
    ax2.set_title('Bird Velocity Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Velocity (pixels/s)')
    ax2.grid(True, alpha=0.3)
    ax2.plot(relative_times, x_vel_data, 'b-', label='X Velocity', linewidth=2)
    ax2.plot(relative_times, y_vel_data, 'r-', label='Y Velocity', linewidth=2)
    ax2.legend()
    
    # Trajectory chart
    ax3.set_title('Bird Trajectory', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Position (pixels)')
    ax3.set_ylabel('Y Position (pixels)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 960)
    ax3.set_ylim(0, 600)
    ax3.invert_yaxis()  # Invert Y axis so ground is at bottom
    
    # Draw ground line
    ax3.axhline(y=500, color='green', linestyle='-', alpha=0.7, label='Ground')
    
    # Plot trajectory
    ax3.plot(x_pos_data, y_pos_data, 'b-', linewidth=2, alpha=0.8)
    ax3.scatter(x_pos_data[-1], y_pos_data[-1], color='red', s=100, zorder=5, label='Current Position')
    ax3.legend()
    
    plt.tight_layout()
    
    # Update the display
    plt.draw()
    plt.pause(0.001)  # Small pause to allow GUI updates


def create_shot_table(shot_history):
    """Create a matplotlib table with shot data"""
    if not shot_history:
        return None
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Shot History Table', fontsize=16, fontweight='bold')
    
    # Prepare data for table
    shot_numbers = [shot.get('shot_number', '?') for shot in shot_history[-10:]]  # Last 10 shots
    angles = [f"{shot.get('angle_deg', 0):.1f}°" for shot in shot_history[-10:]]
    forces = [f"{shot.get('force', 0):.0f}" for shot in shot_history[-10:]]
    hits = ["HIT" if shot.get('hit_target', False) else "MISS" for shot in shot_history[-10:]]
    
    # Create table
    table_data = [shot_numbers, angles, forces, hits]
    column_labels = ['Shot#', 'Angle', 'Force', 'Result']
    
    table = ax.table(cellText=list(zip(*table_data)), 
                     colLabels=column_labels,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color code the hit/miss cells
    for i, hit in enumerate(hits):
        if hit == "HIT":
            table[(i+1, 3)].set_facecolor('lightgreen')
        else:
            table[(i+1, 3)].set_facecolor('lightcoral')
    
    # Add statistics below table
    total_shots = len(shot_history)
    hits_count = sum(1 for shot in shot_history if shot.get('hit_target', False))
    hit_rate = (hits_count / total_shots) * 100 if total_shots > 0 else 0
    
    stats_text = f"Statistics: {hits_count}/{total_shots} hits ({hit_rate:.1f}% success rate)"
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def main() -> None:
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

        # Chart management
        show_charts = False  # Start with charts hidden
        chart_window_open = False
        
        # Data for charts
        time_data = deque(maxlen=300)  # Store last 5 seconds at 60 FPS
        x_pos_data = deque(maxlen=300)
        y_pos_data = deque(maxlen=300)
        x_vel_data = deque(maxlen=300)
        y_vel_data = deque(maxlen=300)
        
        # Initialize matplotlib for interactive mode
        plt.ion()  # Turn on interactive mode
        fig = None
        ax1 = None
        ax2 = None
        ani = None

        # Shot tracking for table view
        shot_history = []  # List of shot data dictionaries
        current_shot_data = {}  # Data for current shot in progress
        
         # Table view state
        show_table = False  # Start with table hidden
        table_window_open = False  # Track if matplotlib table window is open
        table_fig = None  # Reference to table figure
        table_font = pygame.font.Font(None, 24)  # Smaller font for table

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if shots_fired < max_shots and bird.body.position.x < 500:
                        start_pos = pygame.mouse.get_pos()
                        launching = True
                        # Start tracking shot data
                        current_shot_data = {
                            'shot_number': shots_fired + 1,
                            'start_time': time.time(),
                            'start_pos': start_pos
                        }

                elif event.type == pygame.MOUSEBUTTONUP and launching:
                    end_pos = pygame.mouse.get_pos()

                    # Calculate launch direction: from start to end (opposite of current)
                    dx = start_pos[0] - end_pos[0]  # Reverse X direction (left drag = right launch)
                    dy = start_pos[1] - end_pos[1]  # Reverse Y direction (up drag = down launch)
                    velocity_multiplier = 3
                    velocity = (dx * velocity_multiplier, dy * velocity_multiplier)
                    bird.body.apply_impulse_at_local_point(velocity)
                        
                    # Calculate shot parameters for table
                    drag_distance = math.sqrt(dx**2 + dy**2)
                    force = drag_distance * velocity_multiplier  # Convert drag distance to force
                    angle_rad = math.atan2(-dy, dx)  # Calculate angle (negative dy because y increases downward)
                    angle_deg = math.degrees(angle_rad)
                    
                    # Complete shot data
                    current_shot_data.update({
                        'end_pos': end_pos,
                        'velocity': velocity,
                        'force': force,
                        'angle_deg': angle_deg,
                        'drag_distance': drag_distance,
                        'hit_target': False  # Will be updated when hit detection occurs
                    })
                    shots_fired += 1
                    launching = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset button
                        bird = reset_game(bird, target, space)
                        score = 0
                        shots_fired = 0
                        # Clear chart data on reset
                        time_data.clear()
                        x_pos_data.clear()
                        y_pos_data.clear()
                        x_vel_data.clear()
                        y_vel_data.clear()
                    elif event.key == pygame.K_c:  # Toggle charts
                        show_charts = not show_charts
                        if show_charts and not chart_window_open:
                            # Create chart window
                            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
                            fig.suptitle('Bird Physics Data', fontsize=16, fontweight='bold')
                            chart_window_open = True
                        elif not show_charts and chart_window_open:
                            # Close chart window
                            plt.close(fig)
                            chart_window_open = False
                            fig = None
                            ax1 = None
                            ax2 = None
                            ax3 = None
                    elif event.key == pygame.K_t:  # Toggle table
                        show_table = not show_table
                        if show_table and not table_window_open:
                            # Create table window
                            table_fig = create_shot_table(shot_history)
                            if table_fig:
                                table_window_open = True
                        elif not show_table and table_window_open:
                            # Close table window
                            if table_fig:
                                plt.close(table_fig)
                                table_window_open = False
                                table_fig = None

            # Step physics simulation (60 FPS)
            space.step(1/60)
            
            # Collect data for charts
            current_time = time.time()
            bird_pos = bird.body.position
            bird_vel = bird.body.velocity
            
            time_data.append(current_time)
            x_pos_data.append(bird_pos.x)
            y_pos_data.append(bird_pos.y)
            x_vel_data.append(bird_vel.x)
            y_vel_data.append(bird_vel.y)
            
            # Update charts every 15 frames for performance
            if show_charts and chart_window_open and len(time_data) % 15 == 0:
                update_charts(fig, ax1, ax2, ax3, time_data, x_pos_data, y_pos_data, x_vel_data, y_vel_data)
            
            # Check for target hit
            if check_target_hit(bird, target):
                score += 100
                # Update shot data if we have current shot data
                if current_shot_data:
                    current_shot_data['hit_target'] = True
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                    # Refresh table if it's open
                    if show_table and table_window_open and table_fig:
                        plt.close(table_fig)
                        table_fig = create_shot_table(shot_history)
                reset_target(target)
                bird = reset_bird(space, bird)
                shots_fired = 0  # Reset shots for new target

            # Reset bird if it's moving very slowly (landed)
            if (abs(bird.body.velocity.x) < 5 and 
                abs(bird.body.velocity.y) < 5 and 
                shots_fired > 0 and
                bird.body.position.x > 500):
                # Finalize shot data if bird landed without hitting target
                if current_shot_data and not current_shot_data.get('hit_target', False):
                    current_shot_data['hit_target'] = False
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                bird = reset_bird(space, bird)                

            if bird.body.position.x > 960 or bird.body.position.x < 100:
                if current_shot_data and not current_shot_data.get('hit_target', False):
                    current_shot_data['hit_target'] = False
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                bird = reset_bird(space, bird)

            # Render everything
            render_game(screen, space, bird, target, launching, start_pos, score, shots_fired, max_shots, font, width, height, show_charts, show_table)

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
    return distance < 35  # Bird radius + half target size


def render_game(screen, space, bird, target, launching, start_pos, score, shots_fired, max_shots, font, width, height, show_charts, show_table):
    """Render the entire game"""
    # Draw solid light blue background
    screen.fill((173, 216, 230))
    
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
    
    # Chart toggle indicator
    chart_status = "ON" if show_charts else "OFF"
    chart_text = font.render(f"Charts: {chart_status} (Press C to toggle)", True, (0, 255, 0) if show_charts else (255, 0, 0))
    screen.blit(chart_text, (20, 140))

    # Table toggle indicator
    table_status = "ON" if show_table else "OFF"
    table_text = font.render(f"Table: {table_status} (Press T to toggle)", True, (0, 255, 0) if show_table else (255, 0, 0))
    screen.blit(table_text, (20, 180))

    # Game over message
    if shots_fired >= max_shots:
        game_over_text = font.render("Game Over! Press R to restart", True, (255, 0, 0))
        screen.blit(game_over_text, (width//2 - 150, height//2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
