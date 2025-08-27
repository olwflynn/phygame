## Angry-Birds Style Physics Game – Plan

### Scope
- **Core loop**: Player gets 3 shots to hit the pig. After each player shot, the AI suggests its best shot (angle/force) and then takes its own shot.
- **Physics**: Use `pymunk` for world simulation with ground, slingshot, bird (circle), target pig (box).
- **Visualization**: Use `pygame` to render world, trajectories, and the AI “best shot” indicator line.
- **Scoring**: Closest distance to target on a shot, hit detection, first-to-hit wins the round. Track overall score across attempts.

### Tech
- **Engine**: Python 3.10+; `pymunk` for physics; `pygame` for rendering.
- **AI**: Monte Carlo simulation of angle/force samples to estimate best expected distance/hit probability.
- **Tests**: `pytest` for unit tests; a minimal e2e that boots the game loop headless and steps a few frames.

### Milestones / TODOs
1. [x] Create plan.md with scope, milestones, and open questions
2. [ ] Scaffold project structure and minimal runnable skeleton
3. [ ] Initialize git repo and make initial commit
4. [ ] Add unit and e2e test skeletons
5. [ ] Define AI module interface for Monte Carlo shooter
6. [ ] Add requirements and minimal README
7. [ ] Collect answers to open questions from user

### Architecture (initial)
- `src/main.py`: Entry-point; sets up game loop (pygame), creates world and entities, routes input for angle/force, runs turns.
- `src/game/config.py`: Constants (window size, gravity, time step, colors, slingshot position, limits on angle/force).
- `src/game/entities.py`: Entity creation helpers for ground, bird, target; and reset logic.
- `src/game/physics.py`: Pymunk world setup, step/update functions, collision handling (hit detection).
- `src/game/ai.py`: AI interface and Monte Carlo placeholder with dependency on physics step for rollout simulations.
- `tests/`: unit tests for physics setup and basic entity placement; e2e smoke test.

### Gameplay Details (initial draft)
- Shots per side: Player gets 3 shots per round; AI mirrors with 3 shots (alternating after each player shot).
- Controls: Adjust angle (degrees) and force (scalar) via keys; press space to shoot. Minimal UI showing current angle/force.
- AI: After the player’s shot resolves (bird comes to rest or timeout), run MC sampling (e.g., 200 samples within limits) to propose best angle/force. Draw a line from slingshot showing AI’s suggestion.
- Scoring: Hit = collision with target body. If no hits, compare minimum distance of any shot trajectory to target center. First hit wins; otherwise closest wins.
- End-of-round UX: Show a banner with updated score (AI vs Player), then reset to next round.

### Open Questions
1. How large should the window be? Proposal: 960x540.
2. Acceptable FPS/time-step? Proposal: 60 FPS, fixed dt.
3. Angle range? Proposal: 5°–85°; Force range? Proposal: 200–1200 (tune).
4. Level size and units: Use pixels≈points with `pymunk` default scaling. OK?
5. Do you want wind or randomness? Default: off.
6. AI compute budget: How many samples per turn? Proposal: 200–500.
7. RESOLVED: End-of-round UX = banner with updated score, then reset to next round.

### Risks
- Tuning physics so shots feel satisfying; aligning force units with `pymunk` masses and gravity.
- Running MC inside frame budget; may need to do it between turns or in slices.

### Next Steps
- Implement minimal skeleton with config, physics world, and a dummy loop that renders ground and target.
- Add AI interface with a stub that returns a fixed angle/force to visualize the suggestion line.

