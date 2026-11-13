"""pairplan: plan pairwise comparisons for ranking systems from human votes."""

from .model import AdjacentPair, Fit, FitError, fit
from .votes import Counts, Vote, VoteError, read_votes, systems_of

__version__ = "0.1.0"

__all__ = [
    "AdjacentPair",
    "Counts",
    "Fit",
    "FitError",
    "Vote",
    "VoteError",
    "__version__",
    "fit",
    "read_votes",
    "systems_of",
]
