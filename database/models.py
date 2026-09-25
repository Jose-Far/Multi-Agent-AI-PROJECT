"""
database/models.py — Data Models for Investigation Persistence
==============================================================

Day 16 — Database & Investigation History

Defines dataclasses for investigations, agent contributions,
risk factors, and evidence records.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class AgentResultRecord:
    agent_name: str
    display_key: str
    status: str
    signal_available: bool
    prediction: Optional[str] = None
    risk_score: Optional[float] = None
    raw_risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    calibration_curve: Optional[str] = None
    operational_meaning: Optional[str] = None


@dataclass
class RiskFactorRecord:
    factor: str
    agent_name: Optional[str] = None
    severity: Optional[str] = None


@dataclass
class EvidenceRecord:
    description: str
    agent_name: Optional[str] = None
    evidence_type: Optional[str] = None
    severity: Optional[str] = None


@dataclass
class AnalysisRecord:
    analysis_id: str
    url: str
    created_at: str
    status: str
    final_verdict: str
    usable_agent_count: int
    total_agent_count: int
    coverage: float
    consensus_available: bool
    consensus_satisfied: bool
    conflict_detected: bool
    limited_evidence: bool
    domain: Optional[str] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    evidence_state: Optional[str] = None
    decision_basis_json: Optional[str] = None
    risk_calibration_json: Optional[str] = None
    execution_time_ms: Optional[float] = None
    notes: Optional[str] = None
    tags: Optional[str] = None
    agent_results: List[AgentResultRecord] = field(default_factory=list)
    risk_factors: List[RiskFactorRecord] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)
