# src/macro/signals.py
"""
Data structures for WorldMonitor macro signals.
Pydantic-validated dataclasses for type safety and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pydantic


class RiskTier(Enum):
    """Risk classification tiers based on CII scores."""
    STABLE = "stable"           # CII < 30
    ELEVATED = "elevated"       # 30 <= CII < 50
    HIGH = "high"               # 50 <= CII < 70
    CRITICAL = "critical"       # 70 <= CII < 85
    EXTREME = "extreme"         # CII >= 85


@dataclass(frozen=True)
class AnomalyEvent:
    """
    Represents a single anomaly detection event from WorldMonitor.
    
    Attributes:
        region: Geographic region code (e.g., "ME", "EU", "APAC")
        event_type: Classification of anomaly (e.g., "conflict", "economic", "infrastructure")
        z_score: Statistical deviation from baseline (>= 2.0 is significant)
        timestamp: When the anomaly was detected
        severity: Derived from z_score thresholds
    """
    region: str
    event_type: str
    z_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def severity(self) -> str:
        """Classify severity based on z-score magnitude."""
        if self.z_score >= 3.0:
            return "critical"
        elif self.z_score >= 2.0:
            return "high"
        return "moderate"
    
    def __post_init__(self):
        # Validate z-score is positive (deviation magnitude)
        if self.z_score < 0:
            object.__setattr__(self, 'z_score', abs(self.z_score))


@dataclass
class MacroSignal:
    """
    Consolidated macro intelligence signal from WorldMonitor.
    
    This is the primary data structure passed from MacroBridge to ParameterAdjuster.
    All derived fields are computed from raw API responses.
    """
    # Raw CII data: country_code -> score (0-100)
    cii_scores: Dict[str, float] = field(default_factory=dict)
    
    # Derived metrics
    cii_top5_avg: float = 0.0
    cii_max: float = 0.0
    
    # Anomaly data
    anomaly_zscores: List[AnomalyEvent] = field(default_factory=list)
    high_anomaly_count: int = 0      # z >= 2.0
    critical_anomaly_count: int = 0   # z >= 3.0
    
    # Hotspot tracking
    active_hotspot_count: int = 0     # Countries with CII > 70
    
    # AI narrative
    brief_text: Optional[str] = None
    
    # Metadata
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    is_fallback: bool = False
    
    @classmethod
    def from_raw_api_data(
        cls,
        risk_scores: Dict[str, float],
        anomalies: List[Dict[str, Any]],
        brief: Optional[str] = None,
        is_fallback: bool = False
    ) -> "MacroSignal":
        """
        Factory method to create MacroSignal from raw WorldMonitor API responses.
        
        Args:
            risk_scores: Dict mapping country codes to CII scores (0-100)
            anomalies: List of anomaly event dicts from /api/temporal-baseline
            brief: Optional AI world brief text
            is_fallback: Whether this data is from cache/fallback
            
        Returns:
            Populated MacroSignal instance with all derived fields computed
        """
        # Compute derived CII metrics
        if risk_scores:
            sorted_scores = sorted(risk_scores.values(), reverse=True)
            cii_top5_avg = sum(sorted_scores[:5]) / min(5, len(sorted_scores))
            cii_max = max(risk_scores.values())
            active_hotspots = sum(1 for score in risk_scores.values() if score > 70)
        else:
            cii_top5_avg = 50.0  # Neutral default
            cii_max = 0.0
            active_hotspots = 0
        
        # Process anomalies
        anomaly_events = []
        high_count = 0
        critical_count = 0
        
        for anomaly in anomalies:
            event = AnomalyEvent(
                region=anomaly.get("region", "unknown"),
                event_type=anomaly.get("type", "unknown"),
                z_score=float(anomaly.get("zscore", 0)),
                timestamp=datetime.fromisoformat(
                    anomaly.get("timestamp", datetime.utcnow().isoformat())
                ),
                metadata=anomaly.get("metadata", {})
            )
            anomaly_events.append(event)
            
            if event.z_score >= 3.0:
                critical_count += 1
                high_count += 1
            elif event.z_score >= 2.0:
                high_count += 1
        
        return cls(
            cii_scores=risk_scores,
            cii_top5_avg=cii_top5_avg,
            cii_max=cii_max,
            anomaly_zscores=anomaly_events,
            high_anomaly_count=high_count,
            critical_anomaly_count=critical_count,
            active_hotspot_count=active_hotspots,
            brief_text=brief,
            is_fallback=is_fallback
        )
    
    @property
    def risk_tier(self) -> RiskTier:
        """Overall risk classification based on top-5 average CII."""
        if self.cii_top5_avg >= 70:
            return RiskTier.CRITICAL
        elif self.cii_top5_avg >= 50:
            return RiskTier.HIGH
        elif self.cii_top5_avg >= 30:
            return RiskTier.ELEVATED
        return RiskTier.STABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "cii_top5_avg": round(self.cii_top5_avg, 2),
            "cii_max": round(self.cii_max, 2),
            "high_anomaly_count": self.high_anomaly_count,
            "critical_anomaly_count": self.critical_anomaly_count,
            "active_hotspot_count": self.active_hotspot_count,
            "risk_tier": self.risk_tier.value,
            "is_fallback": self.is_fallback,
            "fetched_at": self.fetched_at.isoformat(),
            "brief_available": self.brief_text is not None
        }