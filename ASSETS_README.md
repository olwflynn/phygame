# Adding Real Images to PhyGame

## Current Placeholders
The game currently uses simple colored shapes as placeholders:
- **Bird**: Orange circle with red border
- **House**: Simple house made of rectangles and triangles
- **Background**: Gradient sky, clouds, trees, grass

## How to Add Real Images

### 1. Prepare Your Images
- **Bird**: Small JPG/PNG (recommended: 28x28 pixels or 32x32 pixels)
- **House**: JPG/PNG (recommended: 80x80 pixels or similar)
- Place them in the `assets/` folder

### 2. Load Images in main.py
Replace the placeholder drawing code with image loading:

```python
# At the top of main.py, after pygame.init()
bird_image = pygame.image.load("assets/bird.jpg")
house_image = pygame.image.load("assets/house.jpg")

# Scale images if needed
bird_image = pygame.transform.scale(bird_image, (28, 28))
house_image = pygame.transform.scale(house_image, (80, 80))
```

### 3. Replace Drawing Code
In the `render_game` function:

```python
# Replace bird drawing:
# pygame.draw.circle(screen, (255, 165, 0), (int(bird_pos.x), int(bird_pos.y)), 14)
bird_rect = bird_image.get_rect(center=(int(bird_pos.x), int(bird_pos.y)))
screen.blit(bird_image, bird_rect)

# Replace house drawing:
# (all the house drawing code)
house_rect = house_image.get_rect(center=(int(target_pos.x), int(target_pos.y)))
screen.blit(house_image, house_rect)
```

## Image Requirements
- **Format**: JPG, PNG, or GIF
- **Size**: Keep reasonable (under 100x100 pixels for performance)
- **Transparency**: PNG with alpha channel works best for irregular shapes
- **Style**: Cartoon/2D style fits the game aesthetic

## Performance Tips
- Load images once at startup, not every frame
- Use `pygame.transform.scale()` to resize images
- Consider using sprite sheets for multiple images
- Keep image file sizes small (< 100KB each)
