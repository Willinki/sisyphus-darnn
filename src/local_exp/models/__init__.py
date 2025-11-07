import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)


def _autodiscover():
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:  # optionally recurse if you use subpackages
            continue
        if m.name in {"registry"}:
            continue
        logger.info(f"Discovering models in {m.name}")
        importlib.import_module(f"{__name__}.{m.name}")


_autodiscover()
