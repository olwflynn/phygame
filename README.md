# PhyGame

Angry Birds-style physics game with AI-powered shot suggestions, real-time analytics, and dynamic level generation.

## Features

### 🎮 Core Gameplay
- **Physics-based gameplay** using Pygame and Pymunk
- **Drag-and-shoot mechanics** - click and drag to aim and launch the bird
- **Realistic physics simulation** with gravity, collision detection, and momentum
- **Multiple shots per level** (3 shots maximum)
- **Score tracking** with 100 points per target hit

### 🤖 AI-Powered Shot Suggestions
- **Monte Carlo simulation** that tests 1000+ random shot combinations
- **Real-time AI analysis** of angle and force parameters
- **Visual shot suggestions** displayed on screen
- **Interactive simulation** with progress tracking
- **Scatter plot visualization** showing successful vs failed shots

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

- **Monte Carlo Method**: Tests random combinations of angles (0-90°) and forces (100-1200)
- **Real-time Progress**: Shows simulation progress with sample count
- **Best Shot Selection**: Automatically finds the most recent successful shot
- **Visual Feedback**: Displays suggested angle and force on screen
- **Performance Optimization**: Runs 10 samples per frame for smooth gameplay

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

## Technical Details

### Architecture
- **Game Engine**: Pygame for rendering and input handling
- **Physics Engine**: Pymunk for realistic physics simulation
- **AI System**: Custom Monte Carlo implementation
- **Analytics**: Matplotlib for real-time data visualization
- **Level Generation**: OpenAI API integration with fallback systems

### Performance
- **60 FPS** physics simulation
- **Real-time AI** with 10 samples per frame
- **Optimized rendering** with efficient collision detection
- **Memory management** for long gameplay sessions

### Dependencies
- `pygame==2.5.2` - Game engine and rendering
- `pymunk==6.6.0` - Physics simulation
- `matplotlib==3.8.2` - Data visualization
- `openai>=1.102.0` - LLM level generation
- `python-dotenv==1.0.0` - Environment variable management
- `pytest==8.3.2` - Testing framework

## Development

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_physics.py
pytest tests/test_ai.py
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
- Machine learning models
- Advanced optimization algorithms
- Multi-objective scoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.
