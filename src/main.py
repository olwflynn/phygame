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
from .game.ai import SimulationState, start_ai_simulation, update_ai_simulation, stop_ai_simulation
from .game.ui import update_charts, create_shot_table, render_game
from .game.game_state import reset_bird, reset_target, create_shot_data, update_shot_data, finalize_shot_data, restart_simulation
from .game.physics import calculate_launch_parameters, is_bird_landed, is_bird_out_of_bounds, is_bird_on_ground, apply_ground_friction
from .game.levels import load_level, load_next_level
# Add imports for AI mode
from .game.train_rl import RLPolicy, get_action, load_model

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
        show_congrats = False  # Track if we should show congratulations message
        congrats_timer = 0  # Timer for congratulations message

        # UI state
        show_settings = False  # Settings tab visibility

        # AI simulation state
        ai_sim_state = SimulationState()
        show_suggestion = False  # Whether to show AI suggestion
        current_suggestion = None  # Current AI suggestion

        # AI Mode state
        ai_mode = False  # Whether AI mode is active
        ai_policy = None  # The loaded RL policy
        ai_shot_timer = 0  # Timer for AI shot delay
        ai_shot_delay = 60  # Frames to wait before AI takes shot (1 second at 60 FPS)
        ai_thinking = False  # Whether AI is currently "thinking"
        ai_shot_count = 0  # Track AI shots for display

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
        suggestion_font = pygame.font.Font(None, 28)  # Font for suggestion display

        def load_ai_policy():
            """Load the trained RL policy"""
            try:
                policy = load_model("rl_policy.pth")
                policy.eval()  # Set to evaluation mode
                print("AI Policy loaded successfully!")
                return policy
            except Exception as e:
                print(f"Failed to load AI policy: {e}")
                return None

        def ai_take_shot():
            """Make the AI take a shot using the loaded policy"""
            nonlocal shots_fired, ai_shot_count, current_shot_data  # Declare nonlocal variables
            
            if ai_policy is None or shots_fired >= max_shots or episode_over:
                return False
            
            try:
                # Get current game state
                state = (space, bird, targets, obstacles)
                
                # Get action from policy
                action_dict = get_action(ai_policy, state)
                angle_deg, impulse_magnitude = action_dict["samples"]
                
                # Convert to velocity vector
                angle_rad = math.radians(angle_deg.item())
                vx = impulse_magnitude.item() * math.cos(angle_rad)
                vy = -impulse_magnitude.item() * math.sin(angle_rad)  # Negative because y increases downward
                velocity = (vx, vy)
                
                # Apply the shot
                bird.body.apply_impulse_at_local_point(velocity)
                
                # Update shot data
                current_shot_data = create_shot_data(shots_fired + 1, time.time(), bird.body.position)
                current_shot_data = update_shot_data(
                    current_shot_data, 
                    bird.body.position,  # End position same as start for AI
                    velocity, 
                    impulse_magnitude.item(), 
                    angle_deg.item(), 
                    0.0  # No drag distance for AI
                )
                shots_fired += 1
                ai_shot_count += 1
                
                print(f"AI Shot {ai_shot_count}: Angle={angle_deg.item():.1f}°, Impulse={impulse_magnitude.item():.1f}")
                
                return True
                
            except Exception as e:
                print(f"AI shot failed: {e}")
                return False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Only allow mouse interactions if not in AI mode
                    if not ai_mode and shots_fired < max_shots and bird.body.position.x < 500 and not episode_over and not show_settings:
                        start_pos = pygame.mouse.get_pos()
                        launching = True
                        # Start tracking shot data
                        current_shot_data = create_shot_data(shots_fired + 1, time.time(), start_pos)

                elif event.type == pygame.MOUSEBUTTONUP and launching and not ai_mode:
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
                    if event.key == pygame.K_ESCAPE:  # Close settings
                        show_settings = False
                    elif event.key == pygame.K_TAB:  # Toggle settings
                        show_settings = not show_settings
                    elif event.key == pygame.K_r:  # Reset button
                        bird = reset_bird(space, bird, bird_start_pos)
                        target = reset_target(space, target, target_start_pos)
                        score = 0
                        shots_fired = 0
                        episode_over = False  # Reset episode state
                        ai_shot_count = 0  # Reset AI shot count
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
                            print(f"Number of bodies in space: {len(space.bodies)}")
                            
                            bird_start_pos = bird.body.position
                            target_start_pos = target.body.position
                            
                            # Reset game state for new level
                            score = 0
                            shots_fired = 0
                            episode_over = False
                            level_number += 1
                            ai_shot_count = 0  # Reset AI shot count
                            
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
                        current_level_type = "LLM" if use_llm_for_levels else "Random"
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
                    elif event.key == pygame.K_s:  # Toggle AI simulation
                        if not ai_sim_state.running and not show_suggestion:
                            # Start AI simulation
                            start_ai_simulation(space, target, ai_sim_state, plot=True)
                            print("AI simulation started")
                        elif ai_sim_state.running:
                            # Stop AI simulation immediately
                            stop_ai_simulation(ai_sim_state)
                            print("AI simulation stopped - returning to game")
                        elif show_suggestion:
                            # Hide suggestion
                            show_suggestion = False
                            current_suggestion = None
                    elif event.key == pygame.K_a:  # Toggle AI Mode
                        if not ai_mode:
                            # Enable AI mode
                            ai_policy = load_ai_policy()
                            if ai_policy is not None:
                                ai_mode = True
                                ai_thinking = True
                                ai_shot_timer = 0
                                ai_shot_count = 0
                                print("AI Mode enabled - AI will take over!")
                            else:
                                print("Failed to load AI policy - AI mode not enabled")
                        else:
                            # Disable AI mode
                            ai_mode = False
                            ai_thinking = False
                            ai_shot_timer = 0
                            print("AI Mode disabled - Manual control restored")

            # AI Mode logic
            if ai_mode and ai_policy is not None:
                # Check if bird is ready for next shot (not moving and in starting position)
                bird_vel = bird.body.velocity
                bird_pos = bird.body.position
                
                # If bird is stationary and in starting area, start thinking about next shot
                if (abs(bird_vel.x) < 1 and abs(bird_vel.y) < 1 and 
                    bird_pos.x < 200 and shots_fired < max_shots and not episode_over):
                    
                    if not ai_thinking:
                        ai_thinking = True
                        ai_shot_timer = 0
                        print("AI is thinking...")
                    
                    ai_shot_timer += 1
                    
                    # Take shot after delay
                    if ai_shot_timer >= ai_shot_delay:
                        if ai_take_shot():
                            ai_thinking = False
                            ai_shot_timer = 0
                        else:
                            ai_thinking = False
                            ai_shot_timer = 0
                else:
                    ai_thinking = False
                    ai_shot_timer = 0

            # Update AI simulation
            if ai_sim_state.running:
                simulation_complete = update_ai_simulation(space, target, ai_sim_state, plot=True)
                if simulation_complete:
                    if ai_sim_state.best_suggestion and not ai_sim_state.should_stop:
                        current_suggestion = ai_sim_state.best_suggestion
                        show_suggestion = True
                        print("AI simulation completed")
                    else:
                        print("AI simulation stopped early")
            elif not ai_sim_state.running and ai_sim_state.current_sample > 0:
                # Simulation was stopped, reset state to return to game
                ai_sim_state.progress = 0.0
                ai_sim_state.current_sample = 0
                ai_sim_state.results = []
                ai_sim_state.best_suggestion = None
                ai_sim_state.should_stop = False
                print("AI simulation state reset - back to game")

            # Step physics simulation (60 FPS) - only if not in settings
            if not show_settings:
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
                show_congrats = True
                congrats_timer = 60  # Show for 1 seconds at 60 FPS
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
                
                # Go to next level, carry score over, reset birds to max shots=3
                try:
                    # Generate and load new level
                    bird, targets, obstacles = load_next_level(space, use_llm=use_llm_for_levels, prev_bird=bird, prev_target=target)
                    target = targets[0]
                    print(f"Number of bodies in space: {len(space.bodies)}")
                    
                    bird_start_pos = bird.body.position
                    target_start_pos = target.body.position
                    
                    # Reset game state for new level but keep score
                    shots_fired = 0  # Reset shots to 0, max_shots stays 3
                    episode_over = False
                    level_number += 1
                    ai_shot_count = 0  # Reset AI shot count
                    
                    # Clear chart data for new level
                    time_data.clear()
                    x_pos_data.clear()
                    y_pos_data.clear()
                    x_vel_data.clear()
                    y_vel_data.clear()
                    
                    # Clear shot history for new level
                    shot_history.clear()
                    
                    print(f"Level {level_number} completed! Score: {score}. Loading next level...")
                except Exception as e:
                    print(f"Error loading next level: {e}")

            # Apply ground friction when bird is on the floor
            if is_bird_on_ground((bird_pos.x, bird_pos.y)):
                apply_ground_friction(bird)

            # Update congratulations timer
            if show_congrats:
                congrats_timer -= 1
                if congrats_timer <= 0:
                    show_congrats = False

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
            render_game(screen, space, bird, target, obstacles, launching, start_pos, velocity_multiplier, score, shots_fired, max_shots, font, width, height, show_charts, show_table, show_suggestion, current_suggestion, suggestion_font, episode_over, level_number, current_level_type, show_settings, ai_sim_state.progress, ai_sim_state.current_sample, ai_sim_state.total_samples, show_congrats, ai_mode, ai_thinking, ai_shot_count)

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
