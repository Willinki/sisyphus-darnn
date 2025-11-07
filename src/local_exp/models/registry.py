from typing import Callable, Dict

# Global registry mapping names to builder functions
_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model builder function under a given name."""

    def decorator(fn: Callable) -> Callable:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return a sorted list of available models."""
    return sorted(_MODEL_REGISTRY)
