"""Compatibility wrapper for legacy imports.

The policy implementation now lives in modules/species_policy.py. This module keeps
the old import path alive for callers that still do ``import agent_policy``.
"""
from modules.species_policy import MLPSpeciesPolicy as PolicyNet, PolicyManager

__all__ = ["PolicyNet", "PolicyManager"]
