# PhyGame

Angry Birds-style physics game with AI-powered shot suggestions, real-time analytics, dynamic level generation, and reinforcement learning training capabilities.

## Features

### 🎮 Core Gameplay
- **Physics-based gameplay** using Pygame and Pymunk
- **Drag-and-shoot mechanics** - click and drag to aim and launch the bird
- **Realistic physics simulation** with gravity, collision detection, and momentum
- **Multiple shots per level** (3 shots maximum)
- **Score tracking** with 100 points per target hit

### 🤖 AI-Powered Shot Suggestions
- **Monte Carlo simulation** that tests 1000+ random shot combinations
- **Reinforcement Learning policy** trained with REINFORCE algorithm
- **Real-time AI analysis** of angle and force parameters
- **Visual shot suggestions** displayed on screen
- **Interactive simulation** with progress tracking
- **Scatter plot visualization** showing successful vs failed shots

### 🧠 Reinforcement Learning Training
- **REINFORCE policy gradient algorithm** for training shot selection
- **Neural network policy** with 57-dimensional state representation
- **Advanced feature engineering** including obstacle density, target relationships, and physics hints
- **Curriculum learning** with distance-based rewards early in training
- **Real-time training visualization** with 9 comprehensive metrics plots
- **Entropy regularization** for exploration encouragement
- **Learning rate scheduling** and gradient clipping for stable training

### 🏗️ Dynamic Level Generation
- **LLM-powered level creation** using OpenAI GPT-3.5-turbo
- **Predefined level system** with easy, default, and hard configurations
- **Custom level support** via JSON configuration
- **Automatic level progression** with increasing difficulty
- **Fallback system** to default levels if LLM generation fails

### 📊 Real-Time Analytics
- **Live physics charts** showing bird position and velocity over time
- **Shot tracking table** with detailed statistics for each attempt
- **Performance metrics** including angle, force, distance, and success rate
- **Interactive matplotlib visualizations** in separate windows
- **Training metrics dashboard** with reward tracking, loss monitoring, and policy evolution

### 🎨 Visual Features
- **Trajectory preview** while aiming
- **Real-time physics visualization**
- **Settings panel** with toggleable features

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd phygame
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API (optional, for LLM level generation):**
   ```bash
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Running the Game
```bash
python -m src.main
```

### Training the RL Model
```bash
# Train a new model (30,000 episodes)
python -m src.game.train_rl

# Load and use pre-trained model
# The trained model is automatically loaded as rl_policy.pth
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Mouse** | Click and drag to aim and shoot |
| **R** | Reset current level |
| **N** | Load next level |
| **L** | Toggle between LLM and predefined level generation |
| **S** | Start/stop AI shot simulation |
| **C** | Toggle real-time physics charts |
| **T** | Toggle shot tracking table |
| **Tab** | Toggle settings panel |
| **Esc** | Close settings panel |

### Game Mechanics

1. **Aiming**: Click and drag from the bird to set trajectory
2. **Shooting**: Release mouse button to launch with calculated force
3. **Scoring**: Hit the target (house) to score 100 points
4. **Level Progression**: Use 'N' to generate new levels
5. **AI Assistance**: Press 'S' to get AI-powered shot suggestions

### AI Simulation Features

#### Monte Carlo Method
- Tests random combinations of angles (0-90°) and forces (100-1200)
- Real-time progress showing simulation progress with sample count
- Best shot selection automatically finds the most recent successful shot
- Visual feedback displays suggested angle and force on screen
- Performance optimization runs 10 samples per frame for smooth gameplay

#### Reinforcement Learning Policy
- **Neural Network Architecture**: 3-layer MLP (57 → 64 → 64 → 4)
- **State Representation**: 57 features including bird/target positions, obstacle configurations, physics hints
- **Action Space**: Continuous angle (0-85°) and impulse magnitude (100-2000)
- **Training Algorithm**: REINFORCE with entropy regularization
- **Curriculum Learning**: Distance-based rewards for first 20,000 episodes
- **Hit Rate**: Achieves ~30% success rate after 30,000 episodes

### Level Generation Options

#### LLM-Generated Levels (Default)
- Uses OpenAI GPT-3.5-turbo to create unique, challenging levels
- Requires `OPENAI_API_KEY` in `.env` file
- Generates 4-10 obstacles with strategic placement
- Ensures targets are reachable with game physics

#### Predefined Levels
- **Easy**: Simple layout with minimal obstacles
- **Default**: Balanced challenge with moderate obstacles  
- **Hard**: Complex layout with multiple strategic obstacles

### Analytics and Tracking

#### Real-Time Physics Charts
- **Position tracking**: X and Y coordinates over time
- **Velocity analysis**: Speed and direction changes
- **Performance monitoring**: Frame-by-frame physics data

#### Shot History Table
- **Shot number**: Sequential tracking of attempts
- **Timing data**: Start time and duration of each shot
- **Launch parameters**: Start/end positions, velocity, angle, force
- **Success tracking**: Whether target was hit
- **Distance calculations**: Total distance traveled

#### RL Training Metrics
- **Reward tracking**: Episode rewards and rolling averages
- **Loss monitoring**: Policy gradient loss over time
- **Action distribution**: Angle and impulse parameter evolution
- **Policy statistics**: Mean and standard deviation of policy outputs
- **Hit rate analysis**: Success rate tracking over training
- **Distance metrics**: Average distance to target over time

## Technical Details

### Architecture
- **Game Engine**: Pygame for rendering and input handling
- **Physics Engine**: Pymunk for realistic physics simulation
- **AI System**: Custom Monte Carlo implementation + PyTorch RL policy
- **Analytics**: Matplotlib for real-time data visualization
- **Level Generation**: OpenAI API integration with fallback systems

### RL Implementation Details
- **Framework**: PyTorch for neural network implementation
- **Policy Network**: Outputs mean and log standard deviation for angle and force
- **State Features**: 57-dimensional vector including:
  - Bird and target positions (normalized)
  - Relative target information (distance, angle)
  - Obstacle configurations and density metrics
  - Target-obstacle relationships and blocking analysis
  - Physics hints (optimal angle estimates, distance-based impulse)
- **Training Optimizations**: Adam optimizer with learning rate scheduling, gradient clipping

### Performance
- **60 FPS** physics simulation
- **Real-time AI** with 10 samples per frame
- **Optimized rendering** with efficient collision detection
- **Memory management** for long gameplay sessions
- **GPU acceleration** available for RL training (when PyTorch CUDA is available)

### Dependencies
- `pygame==2.5.2` - Game engine and rendering
- `pymunk==6.6.0` - Physics simulation
- `matplotlib==3.8.2` - Data visualization
- `openai>=1.102.0` - LLM level generation
- `python-dotenv==1.0.0` - Environment variable management
- `pytest==8.3.2` - Testing framework
- `torch==2.5.0` - Neural network training and inference

## Development

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_physics.py
pytest tests/test_ai.py
```

### Training Your Own RL Model
```bash
# Modify training parameters in src/game/train_rl.py
# Key parameters:
# - num_episodes: Number of training episodes (default: 30,000)
# - curriculum_switch: When to switch from distance-based to hit-based rewards
# - learning_rate: Adam optimizer learning rate (default: 0.0005)

python -m src.game.train_rl
```

### Adding Custom Levels
Create JSON files with this structure:
```json
{
  "targets": [
    {"x": 800, "y": 400}
  ],
  "obstacles": [
    {"x": 400, "y": 500, "w": 10, "h": 100}
  ]
}
```

### Extending AI Features
The AI system is modular and can be extended with:
- Different sampling strategies
- Advanced neural network architectures (transformers, attention mechanisms)
- Advantage-based methods (A2C, PPO)
- Multi-objective optimization
- Transfer learning from pre-trained models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.
