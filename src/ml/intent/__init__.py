"""
Intent Classification for Satellite Behavioral Analysis.

Maps maneuver classifications to operational intents with threat context,
using proximity analysis and behavioral pattern detection.
"""

from src.ml.intent.intent_classifier import (
    IntentCategory,
    ThreatLevel,
    IntentResult,
    IntentClassifier,
)
from src.ml.intent.proximity_context import ProximityContext, compute_proximity
from src.ml.intent.threat_escalation import ThreatEscalator
from src.ml.intent.asset_catalog import Asset, AssetCatalog

__all__ = [
    "IntentCategory",
    "ThreatLevel",
    "IntentResult",
    "IntentClassifier",
    "ProximityContext",
    "compute_proximity",
    "ThreatEscalator",
    "Asset",
    "AssetCatalog",
]
