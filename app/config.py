from app.models import DifficultyLevel

DIFFICULTY_CONFIG: dict = {
    DifficultyLevel.easy: {
        "rows": 50,
        "max_steps": 15,
        "dirty": "data/easy_dirty.csv",
        "clean": "data/easy_clean.csv",
    },
    DifficultyLevel.medium: {
        "rows": 200,
        "max_steps": 30,
        "dirty": "data/medium_dirty.csv",
        "clean": "data/medium_clean.csv",
    },
    DifficultyLevel.hard: {
        "rows": 500,
        "max_steps": 60,
        "dirty": "data/hard_dirty.csv",
        "clean": "data/hard_clean.csv",
    },
}
