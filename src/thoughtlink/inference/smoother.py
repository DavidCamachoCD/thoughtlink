"""Re-export MajorityVotingSmoother for backwards compatibility.

The actual implementation lives in confidence.py alongside the other
stability mechanisms.
"""

from thoughtlink.inference.confidence import MajorityVotingSmoother

__all__ = ["MajorityVotingSmoother"]
