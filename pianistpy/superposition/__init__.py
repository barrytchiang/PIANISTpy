"""Linear superposition of models that share one truncated density.

`superpose` combines models built on one truncated potential-density pair
into one model, `moments` returns the velocity moments of the combined model,
and `mixture` combines any per-component quantity by the weights.
"""

__all__ = ["mixture", "moments", "superpose"]

from .core import mixture, moments, superpose
