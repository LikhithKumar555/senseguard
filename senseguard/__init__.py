"""SenseGuard: a camera + radar perception system that notices when its own
sensors are failing, and keeps the car safe when they do."""

__version__ = "0.0.1"

# Subpackages, in the order data flows through the system.
SUBPACKAGES = (
    "data",
    "geometry",
    "faults",
    "perception",
    "health",
    "safety",
    "eval",
)

__all__ = ["__version__", "SUBPACKAGES"]
