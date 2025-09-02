import pygame
import matplotlib.pyplot as plt
import time
import math
from collections import deque
from typing import List, Dict, Any, Optional, Tuple


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


def create_shot_table(shot_history: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """Create a matplotlib table with shot data"""
    if not shot_history:
        return None
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Shot History Table', fontsize=16, fontweight='bold')
    
    # Prepare data for table
    shot_numbers = [shot.get('shot_number', '?') for shot in shot_history[-10:]]  # Last 10 shots
    angles = [f"{shot.get('angle_deg', 0):.1f}°" for shot in shot_history[-10:]]
    impulse_magnitudes = [f"{shot.get('impulse_magnitude', 0):.0f}" for shot in shot_history[-10:]]
    hits = ["HIT" if shot.get('hit_target', False) else "MISS" for shot in shot_history[-10:]]
    
    # Create table
    table_data = [shot_numbers, angles, impulse_magnitudes, hits]
    column_labels = ['Shot#', 'Angle', 'Impulse', 'Result']
    
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


def render_game(screen: pygame.Surface, space, bird, target, obstacles, launching: bool, 
                start_pos: Optional[Tuple[int, int]], velocity_multiplier: float, 
                score: int, shots_fired: int, max_shots: int, font: pygame.font.Font, 
                width: int, height: int, show_charts: bool, show_table: bool, 
                show_suggestion: bool, current_suggestion, suggestion_font: pygame.font.Font,
                episode_over: bool = False, level_number: int = 1, level_type: str = "LLM") -> None:
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
    pygame.draw.rect(screen, (34, 139, 34), (0, 499, 960, 40))  # Grass
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
    
    # Draw obstacles
    for obstacle, size in obstacles:
        obstacle_width, obstacle_height = size
        obstacle_pos = obstacle.body.position

        # Calculate top-left corner (body position is center)
        top_left_x = int(obstacle_pos.x - obstacle_width/2)
        top_left_y = int(obstacle_pos.y - obstacle_height/2)
        pygame.draw.rect(screen, (100, 100, 100), (top_left_x, top_left_y, int(obstacle_width), int(obstacle_height)))

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
        
        # Calculate launch vector (from start_pos to current_pos)
        launch_dx = start_pos[0] - current_pos[0]
        launch_dy = start_pos[1] - current_pos[1]
        drag_distance = math.sqrt(launch_dx**2 + launch_dy**2) / 2
        impulse_magnitude = drag_distance * velocity_multiplier
        # Angle: 0° is to the right, positive is upward (screen y increases downward)
        angle_rad = math.atan2(-launch_dy, launch_dx)
        angle_deg = math.degrees(angle_rad)
        # Draw the angle and impulse magnitude near the current mouse position
        info_text = font.render(f"{angle_deg:.1f}°  {impulse_magnitude:.0f}", True, (255, 255, 0))
        # Offset text a bit from the current mouse position
        text_offset = (30, -30)
        screen.blit(info_text, (current_pos[0] + text_offset[0], current_pos[1] + text_offset[1]))

    # Draw UI elements
    # Score display
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (20, 20))
    
    # Level display
    level_text = font.render(f"Level: {level_number} ({level_type})", True, (255, 255, 255))
    screen.blit(level_text, (20, 60))
    
    # Shots remaining
    shots_text = font.render(f"Shots: {max_shots - shots_fired}/{max_shots}", True, (255, 255, 255))
    screen.blit(shots_text, (20, 100))
    
    # Reset button
    reset_text = font.render("Press R to Reset", True, (255, 0, 0))
    screen.blit(reset_text, (20, 140))
    
    # Next level button
    next_level_text = font.render("Press N for Next Level", True, (0, 255, 255))
    screen.blit(next_level_text, (20, 180))
    
    # Level generation toggle
    level_gen_text = font.render(f"Press L to toggle: {level_type}", True, (255, 255, 0))
    screen.blit(level_gen_text, (20, 220))
    
    # Chart toggle indicator
    chart_status = "ON" if show_charts else "OFF"
    chart_text = font.render(f"Charts: {chart_status} (Press C to toggle)", True, (0, 255, 0) if show_charts else (255, 0, 0))
    screen.blit(chart_text, (20, 260))

    # Table toggle indicator
    table_status = "ON" if show_table else "OFF"
    table_text = font.render(f"Table: {table_status} (Press T to toggle)", True, (0, 255, 0) if show_table else (255, 0, 0))
    screen.blit(table_text, (20, 300))

    # Suggest best shot toggle indicator
    suggestion_status = "ON" if show_suggestion else "OFF"
    suggestion_text = font.render(f"AI Suggestion: {suggestion_status} (Press S to toggle)", True, (0, 255, 0) if show_suggestion else (255, 0, 0))
    screen.blit(suggestion_text, (20, 340))

    # Display AI suggestion if active
    if show_suggestion and current_suggestion:
        # Create a semi-transparent overlay
        overlay = pygame.Surface((400, 200))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (width//2 - 200, height//2 - 100))
        
        # Draw border
        pygame.draw.rect(screen, (255, 255, 255), (width//2 - 200, height//2 - 100, 400, 200), 3)
        
        # Display suggestion text
        title_text = suggestion_font.render("AI Suggestion", True, (255, 255, 255))
        screen.blit(title_text, (width//2 - 80, height//2 - 80))
        
        angle_text = suggestion_font.render(f"Best Angle: {current_suggestion.angle_deg:.1f}°", True, (255, 255, 255))
        screen.blit(angle_text, (width//2 - 100, height//2 - 40))
        
        force_text = suggestion_font.render(f"Best Force: {current_suggestion.impulse_magnitude:.0f}", True, (255, 255, 255))
        screen.blit(force_text, (width//2 - 100, height//2 - 10))
        
        instruction_text = suggestion_font.render("Press S again to close", True, (255, 255, 255))
        screen.blit(instruction_text, (width//2 - 120, height//2 + 30))

    # Game over message - only show when episode is actually over
    if episode_over:
        game_over_text = font.render("Game Over! Press R to restart", True, (255, 0, 0))
        screen.blit(game_over_text, (width//2 - 150, height//2))
