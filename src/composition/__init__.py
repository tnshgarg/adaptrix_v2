"""
Adapter Composition System for Adaptrix.

This module provides revolutionary multi-adapter composition capabilities,
allowing multiple specialized adapters to work together in sophisticated ways.
"""

from .adapter_composer import AdapterComposer, CompositionStrategy
from .attention_mechanisms import AdapterAttention, ConflictResolver
from .composition_strategies import (
    SequentialComposer,
    ParallelComposer, 
    HierarchicalComposer,
    ConditionalComposer
)

__all__ = [
    "AdapterComposer",
    "CompositionStrategy", 
    "AdapterAttention",
    "ConflictResolver",
    "SequentialComposer",
    "ParallelComposer",
    "HierarchicalComposer", 
    "ConditionalComposer"
]
