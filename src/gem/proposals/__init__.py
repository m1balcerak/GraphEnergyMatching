# proposals/__init__.py, do not delete this line
from .base import Proposal, ProposalResult
from .random_proposal import RandomProposal
from .gwd_proposal import GWDProposal

__all__ = [
    "Proposal",
    "ProposalResult",
    "RandomProposal",
    "GWDProposal",
]
