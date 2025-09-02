import os
import sys

import pygame
import pymunk
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .game.entities import create_ground, create_bird, create_world, check_target_hit
from .game.ai import suggest_best_shot, ShotSuggestion
from .game.ui import update_charts, create_shot_table, render_game
from .game.game_state import reset_bird, reset_target, create_shot_data, update_shot_data, finalize_shot_data
from .game.physics import calculate_launch_parameters, is_bird_landed, is_bird_out_of_bounds
from .game.levels import load_level, load_next_level, load_custom_level, load_predefined_level, PREDEFINED_LEVELS
from typing import Optional, Tuple

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
        
        # Game state
        level_number = 1
        use_llm_for_levels = True  # Toggle between LLM and predefined levels
        current_level_type = "LLM"  # Track current level type
        bird, targets, obstacles = load_level(space, use_llm=use_llm_for_levels)
        target = targets[0]
        velocity_multiplier = 5
        bird_start_pos = bird.body.position
        target_start_pos = target.body.position
        
        # Game state
        running = True
        launching = False
        start_pos = None
        score = 0
        shots_fired = 0
        max_shots = 3
        episode_over = False  # Track if episode is actually over

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

        # Suggest best shot state
        show_suggestion = False  # Start with suggestion hidden
        current_suggestion = None  # Current AI suggestion
        suggestion_font = pygame.font.Font(None, 28)  # Font for suggestion display

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if shots_fired < max_shots and bird.body.position.x < 500 and not episode_over:
                        start_pos = pygame.mouse.get_pos()
                        launching = True
                        # Start tracking shot data
                        current_shot_data = create_shot_data(shots_fired + 1, time.time(), start_pos)

                elif event.type == pygame.MOUSEBUTTONUP and launching:
                    end_pos = pygame.mouse.get_pos()

                    # Calculate launch parameters
                    velocity, impulse_magnitude, angle_deg, drag_distance = calculate_launch_parameters(
                        start_pos, end_pos, velocity_multiplier
                    )
                    bird.body.apply_impulse_at_local_point(velocity)
                    
                    # Update shot data
                    current_shot_data = update_shot_data(
                        current_shot_data, end_pos, velocity, impulse_magnitude, angle_deg, drag_distance
                    )
                    shots_fired += 1
                    launching = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset button
                        bird = reset_bird(space, bird, bird_start_pos)
                        target = reset_target(space, target, target_start_pos)
                        score = 0
                        shots_fired = 0
                        episode_over = False  # Reset episode state
                        # Clear chart data on reset
                        time_data.clear()
                        x_pos_data.clear()
                        y_pos_data.clear()
                        x_vel_data.clear()
                        y_vel_data.clear()
                    elif event.key == pygame.K_n:  # Next level button
                        try:
                            # Generate and load new level
                            bird, targets, obstacles = load_next_level(space, use_llm=use_llm_for_levels, prev_bird=bird, prev_target=target)
                            target = targets[0]
                            
                            bird_start_pos = bird.body.position
                            target_start_pos = target.body.position
                            
                            # Reset game state for new level
                            score = 0
                            shots_fired = 0
                            episode_over = False
                            level_number += 1
                            
                            # Clear chart data for new level
                            time_data.clear()
                            x_pos_data.clear()
                            y_pos_data.clear()
                            x_vel_data.clear()
                            y_vel_data.clear()
                            
                            # Clear shot history for new level
                            shot_history.clear()
                            
                            print(f"Loaded level {level_number} ({current_level_type})")
                        except Exception as e:
                            print(f"Error loading next level: {e}")
                    elif event.key == pygame.K_l:  # Toggle level generation method
                        use_llm_for_levels = not use_llm_for_levels
                        current_level_type = "LLM" if use_llm_for_levels else "Predefined"
                        print(f"Level generation switched to: {current_level_type}")
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
                    elif event.key == pygame.K_s:  # Toggle suggest best shot
                        if not show_suggestion:
                            # Get AI suggestion
                            try:
                                current_suggestion = suggest_best_shot(space, plot=True)
                                show_suggestion = True
                            except Exception as e:
                                print(f"Error getting AI suggestion: {e}")
                                current_suggestion = None
                        else:
                            # Hide suggestion
                            show_suggestion = False
                            current_suggestion = None

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
                    current_shot_data = finalize_shot_data(current_shot_data, True)
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                    # Refresh table if it's open
                    if show_table and table_window_open and table_fig:
                        plt.close(table_fig)
                        table_fig = create_shot_table(shot_history)
                target = reset_target(space, target, target_start_pos)
                bird = reset_bird(space, bird, bird_start_pos)
                shots_fired = 0  # Reset shots for new target
                episode_over = False  # Reset episode state

            # Reset bird if it's moving very slowly (landed)
            if (is_bird_landed((bird_vel.x, bird_vel.y)) and 
                bird.body.position.x > 150):
                # Finalize shot data if bird landed without hitting target
                if current_shot_data and not current_shot_data.get('hit_target', False):
                    current_shot_data = finalize_shot_data(current_shot_data, False)
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                bird = reset_bird(space, bird, bird_start_pos)
                
                # Check if episode is over (all shots fired and current shot finished)
                if shots_fired >= max_shots:
                    episode_over = True

            # Reset bird if it goes out of bounds
            if is_bird_out_of_bounds((bird_pos.x, bird_pos.y)):
                if current_shot_data and not current_shot_data.get('hit_target', False):
                    current_shot_data = finalize_shot_data(current_shot_data, False)
                    shot_history.append(current_shot_data.copy())
                    current_shot_data = {}
                bird = reset_bird(space, bird, bird_start_pos)
                
                # Check if episode is over (all shots fired and current shot finished)
                if shots_fired >= max_shots:
                    episode_over = True

            # Render everything
            render_game(screen, space, bird, target, obstacles, launching, start_pos, velocity_multiplier, score, shots_fired, max_shots, font, width, height, show_charts, show_table, show_suggestion, current_suggestion, suggestion_font, episode_over, level_number, current_level_type)

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
