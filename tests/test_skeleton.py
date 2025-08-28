def test_imports():
    import importlib

    # Test that core modules can be imported
    config = importlib.import_module("src.game.config")
    assert config.WINDOW_WIDTH > 0
    
    entities = importlib.import_module("src.game.entities")
    assert hasattr(entities, 'create_bird')
    
    physics = importlib.import_module("src.game.physics")
    assert hasattr(physics, 'create_world')
