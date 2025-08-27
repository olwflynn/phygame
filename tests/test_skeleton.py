def test_imports():
    import importlib

    main = importlib.import_module("src.main")
    assert hasattr(main, "main")

    config = importlib.import_module("src.game.config")
    assert config.WINDOW_WIDTH > 0
