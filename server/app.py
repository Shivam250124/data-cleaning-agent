"""OpenEnv server entry point."""

from app.main import app

# Re-export for OpenEnv compatibility
__all__ = ["app"]
