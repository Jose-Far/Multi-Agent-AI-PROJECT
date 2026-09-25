"""
DNS AI Agent - Feature Schema
==============================

This module defines the canonical feature contract for the DNS Security AI Agent.

Responsibilities
----------------
1. Define the deterministic ML feature order.
2. Normalize raw DNS telemetry.
3. Convert booleans and categorical security policies into numeric values.
4. Validate feature values.
5. Protect the ML pipeline from NaN / infinity / invalid values.
6. Maintain backward-compatible feature aliases where required.
7. Generate a one-row pandas DataFrame suitable for preprocessing/model inference.
8. Provide feature metadata for Explainable AI (SHAP/LIME).
9. Provide schema validation utilities for training and inference.
10. Prevent feature-order mismatches between training and prediction.

Important Security Principle
----------------------------
A DNS feature is an indicator, not proof of phishing.

For example:

    No MX record != Phishing

The DNS Agent should therefore produce structured evidence and risk
signals. Final classification should be performed by the Multi-Agent
Decision Fusion Engine using signals from multiple agents.

Canonical Feature Set
---------------------
The schema intentionally contains 37 features covering:

    - DNS routing and structure
    - Fast-flux / temporal characteristics
    - MX configuration
    - TXT records and payload indicators
    - SPF configuration
    - DMARC configuration
    - Email spoofability

Author: Multi-Agent AI Cybersecurity Analyst
Agent: DNS Security AI Agent
Version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd


# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION = "1.0.0"


# =============================================================================
# FEATURE INFORMATION MODEL
# =============================================================================

@dataclass(frozen=True)
class FeatureInfo:
    """
    Metadata describing a single DNS ML feature.

    Attributes
    ----------
    name:
        Canonical feature name.

    description:
        Human-readable description.

    feature_type:
        Logical feature type:
            - boolean
            - count
            - numeric
            - ordinal

    default:
        Safe default value when the feature is unavailable.

    min_value:
        Minimum expected value.

    max_value:
        Maximum expected value.

    security_relevance:
        Explanation of why the feature matters to DNS security.
    """

    name: str
    description: str
    feature_type: str
    default: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    security_relevance: str = ""


# =============================================================================
# DNS FEATURE SCHEMA
# =============================================================================

class DNSFeatureSchema:
    """
    Canonical schema and normalization engine for the DNS AI Agent.

    This class is intentionally stateless.

    The main pipeline is:

        raw DNS dictionary
                ↓
        normalize_raw_features()
                ↓
        align_and_validate()
                ↓
        pandas DataFrame
                ↓
        preprocessing.py
                ↓
        ML model

    The class should be used by both:

        - trainer.py
        - predictor.py

    This prevents training and inference from accidentally using
    different feature ordering or data types.
    """

    # =========================================================================
    # 1. CANONICAL FEATURE ORDER
    # =========================================================================

    EXPECTED_FEATURE_ORDER: List[str] = [

        # ---------------------------------------------------------------------
        # DNS Routing & Structure
        # ---------------------------------------------------------------------
        "has_a_records",
        "has_aaaa_records",
        "has_cname_record",
        "a_record_count",
        "aaaa_record_count",
        "ns_count",
        "cname_count",
        "total_resolved_ips",
        "has_routing_anomalies",

        # ---------------------------------------------------------------------
        # Temporal / Fast-Flux
        # ---------------------------------------------------------------------
        "min_ttl_value",
        "max_ttl_value",
        "avg_ttl_value",
        "is_fast_flux_candidate",
        "is_active_fast_flux",

        # ---------------------------------------------------------------------
        # MX / Mail Exchange
        # ---------------------------------------------------------------------
        "has_mx_records",
        "mx_record_count",
        "uses_free_mail_provider",
        "uses_disposable_mail_provider",
        "has_suspicious_mx_exchange",
        "lowest_mx_preference",

        # ---------------------------------------------------------------------
        # TXT / DNS Payloads
        # ---------------------------------------------------------------------
        "has_txt_records",
        "txt_record_count",
        "has_domain_verification",
        "has_suspicious_long_txt",
        "has_base64_payload_in_txt",
        "avg_txt_length",

        # ---------------------------------------------------------------------
        # SPF
        # ---------------------------------------------------------------------
        "has_spf_record",
        "spf_record_count",
        "has_multiple_spf_records",
        "spf_includes_count",
        "spf_strictness_score",

        # ---------------------------------------------------------------------
        # DMARC
        # ---------------------------------------------------------------------
        "has_dmarc_record",
        "dmarc_policy_score",
        "dmarc_subdomain_policy_score",

        # ---------------------------------------------------------------------
        # Composite Email Security
        # ---------------------------------------------------------------------
        "is_email_spoofable",
    ]

    # =========================================================================
    # 2. SCHEMA METADATA
    # =========================================================================

    FEATURE_METADATA: Dict[str, FeatureInfo] = {

        # ---------------------------------------------------------------------
        # DNS Routing
        # ---------------------------------------------------------------------

        "has_a_records": FeatureInfo(
            name="has_a_records",
            description="Indicates whether the domain has at least one IPv4 A record.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "A records provide IPv4 routing information for the domain."
            ),
        ),

        "has_aaaa_records": FeatureInfo(
            name="has_aaaa_records",
            description="Indicates whether the domain has at least one IPv6 AAAA record.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "AAAAs indicate IPv6-based routing infrastructure."
            ),
        ),

        "has_cname_record": FeatureInfo(
            name="has_cname_record",
            description="Indicates whether a CNAME record is present.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "CNAME chains can reveal hosting relationships and routing structure."
            ),
        ),

        "a_record_count": FeatureInfo(
            name="a_record_count",
            description="Number of IPv4 A records returned for the domain.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Multiple IPv4 addresses can be legitimate but may also contribute "
                "to infrastructure and fast-flux analysis."
            ),
        ),

        "aaaa_record_count": FeatureInfo(
            name="aaaa_record_count",
            description="Number of IPv6 AAAA records returned for the domain.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides additional information about the domain's routing footprint."
            ),
        ),

        "ns_count": FeatureInfo(
            name="ns_count",
            description="Number of authoritative nameserver records.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Nameserver count contributes to understanding domain infrastructure."
            ),
        ),

        "cname_count": FeatureInfo(
            name="cname_count",
            description="Number of CNAME records detected.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "CNAME relationships can reveal hosting and infrastructure patterns."
            ),
        ),

        "total_resolved_ips": FeatureInfo(
            name="total_resolved_ips",
            description="Total number of unique resolved IP addresses.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Large or rapidly changing IP pools may be relevant to fast-flux analysis."
            ),
        ),

        "has_routing_anomalies": FeatureInfo(
            name="has_routing_anomalies",
            description=(
                "Indicates potentially abnormal routing conditions, such as "
                "missing expected routing endpoints."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Routing anomalies can contribute to infrastructure risk analysis."
            ),
        ),

        # ---------------------------------------------------------------------
        # TTL / Fast Flux
        # ---------------------------------------------------------------------

        "min_ttl_value": FeatureInfo(
            name="min_ttl_value",
            description="Minimum observed DNS TTL value.",
            feature_type="numeric",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Very short TTL values may be associated with rapidly changing DNS "
                "infrastructure, although legitimate services can also use short TTLs."
            ),
        ),

        "max_ttl_value": FeatureInfo(
            name="max_ttl_value",
            description="Maximum observed DNS TTL value.",
            feature_type="numeric",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides context for DNS record lifetime and infrastructure stability."
            ),
        ),

        "avg_ttl_value": FeatureInfo(
            name="avg_ttl_value",
            description="Average observed DNS TTL value.",
            feature_type="numeric",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Average TTL contributes to infrastructure stability and fast-flux analysis."
            ),
        ),

        "is_fast_flux_candidate": FeatureInfo(
            name="is_fast_flux_candidate",
            description=(
                "Indicates that DNS characteristics resemble a possible fast-flux "
                "configuration."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Fast-flux infrastructure can be used to make malicious hosting "
                "more resilient to takedown."
            ),
        ),

        "is_active_fast_flux": FeatureInfo(
            name="is_active_fast_flux",
            description=(
                "Composite indicator based on low TTL values and a high number "
                "of resolved IP addresses."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Potential active fast-flux behavior can indicate suspicious "
                "infrastructure, but should not independently determine phishing."
            ),
        ),

        # ---------------------------------------------------------------------
        # MX
        # ---------------------------------------------------------------------

        "has_mx_records": FeatureInfo(
            name="has_mx_records",
            description="Indicates whether the domain has MX records.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "MX records indicate mail-handling infrastructure for the domain."
            ),
        ),

        "mx_record_count": FeatureInfo(
            name="mx_record_count",
            description="Number of MX records.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides information about the domain's mail infrastructure."
            ),
        ),

        "uses_free_mail_provider": FeatureInfo(
            name="uses_free_mail_provider",
            description=(
                "Indicates whether a detected mail infrastructure pattern "
                "matches a known free-mail provider."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "May provide contextual information about disposable or low-cost "
                "infrastructure, but is not inherently malicious."
            ),
        ),

        "uses_disposable_mail_provider": FeatureInfo(
            name="uses_disposable_mail_provider",
            description=(
                "Indicates whether a disposable or temporary mail service "
                "is associated with the detected configuration."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Disposable mail infrastructure can be relevant to temporary "
                "or abuse-oriented infrastructure."
            ),
        ),

        "has_suspicious_mx_exchange": FeatureInfo(
            name="has_suspicious_mx_exchange",
            description="Indicates a suspicious MX exchange pattern.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Suspicious mail infrastructure may contribute to impersonation "
                "or abuse risk."
            ),
        ),

        "lowest_mx_preference": FeatureInfo(
            name="lowest_mx_preference",
            description="Lowest MX preference value observed.",
            feature_type="numeric",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides additional context about MX routing priority."
            ),
        ),

        # ---------------------------------------------------------------------
        # TXT
        # ---------------------------------------------------------------------

        "has_txt_records": FeatureInfo(
            name="has_txt_records",
            description="Indicates whether TXT records are present.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "TXT records can contain legitimate verification and email "
                "security information as well as anomalous payloads."
            ),
        ),

        "txt_record_count": FeatureInfo(
            name="txt_record_count",
            description="Number of TXT records.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides context about the amount of TXT-based configuration."
            ),
        ),

        "has_domain_verification": FeatureInfo(
            name="has_domain_verification",
            description=(
                "Indicates presence of recognized domain verification "
                "or ownership-verification records."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Verification records can indicate legitimate service integration."
            ),
        ),

        "has_suspicious_long_txt": FeatureInfo(
            name="has_suspicious_long_txt",
            description=(
                "Indicates unusually long TXT content that may warrant inspection."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Unusually long TXT values can sometimes contain encoded or "
                "abnormal payloads."
            ),
        ),

        "has_base64_payload_in_txt": FeatureInfo(
            name="has_base64_payload_in_txt",
            description=(
                "Indicates potential Base64-like encoded content in TXT records."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Encoded TXT content can warrant further investigation, although "
                "encoding alone does not prove malicious activity."
            ),
        ),

        "avg_txt_length": FeatureInfo(
            name="avg_txt_length",
            description="Average length of TXT record content.",
            feature_type="numeric",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Provides context for TXT payload size and configuration complexity."
            ),
        ),

        # ---------------------------------------------------------------------
        # SPF
        # ---------------------------------------------------------------------

        "has_spf_record": FeatureInfo(
            name="has_spf_record",
            description="Indicates whether an SPF record is present.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "SPF helps define which systems are authorized to send email "
                "for a domain."
            ),
        ),

        "spf_record_count": FeatureInfo(
            name="spf_record_count",
            description="Number of SPF records detected.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Multiple SPF records can indicate configuration problems."
            ),
        ),

        "has_multiple_spf_records": FeatureInfo(
            name="has_multiple_spf_records",
            description="Indicates whether multiple SPF records were detected.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Multiple SPF records can cause SPF evaluation problems."
            ),
        ),

        "spf_includes_count": FeatureInfo(
            name="spf_includes_count",
            description="Number of include mechanisms in the SPF record.",
            feature_type="count",
            default=0.0,
            min_value=0.0,
            security_relevance=(
                "Large SPF include chains may increase configuration complexity."
            ),
        ),

        "spf_strictness_score": FeatureInfo(
            name="spf_strictness_score",
            description=(
                "Ordinal representation of SPF policy strictness. "
                "0=None/Invalid, 1=Dangerously Permissive, "
                "2=Permissive, 3=Neutral, 4=Strict."
            ),
            feature_type="ordinal",
            default=0.0,
            min_value=0.0,
            max_value=4.0,
            security_relevance=(
                "Stricter SPF policies can reduce unauthorized email-sending risk."
            ),
        ),

        # ---------------------------------------------------------------------
        # DMARC
        # ---------------------------------------------------------------------

        "has_dmarc_record": FeatureInfo(
            name="has_dmarc_record",
            description="Indicates whether a DMARC record is present.",
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "DMARC provides domain-level email authentication policy enforcement."
            ),
        ),

        "dmarc_policy_score": FeatureInfo(
            name="dmarc_policy_score",
            description=(
                "Ordinal representation of the DMARC policy: "
                "0=Unknown/Invalid, 1=None, 2=Quarantine, 3=Reject."
            ),
            feature_type="ordinal",
            default=0.0,
            min_value=0.0,
            max_value=3.0,
            security_relevance=(
                "Reject provides stronger enforcement against unauthorized "
                "email compared with monitoring-only policies."
            ),
        ),

        "dmarc_subdomain_policy_score": FeatureInfo(
            name="dmarc_subdomain_policy_score",
            description=(
                "Ordinal representation of the DMARC subdomain policy: "
                "0=Unknown/Invalid, 1=None, 2=Quarantine, 3=Reject."
            ),
            feature_type="ordinal",
            default=0.0,
            min_value=0.0,
            max_value=3.0,
            security_relevance=(
                "Subdomain policy influences protection against email spoofing "
                "using subdomains."
            ),
        ),

        # ---------------------------------------------------------------------
        # Composite
        # ---------------------------------------------------------------------

        "is_email_spoofable": FeatureInfo(
            name="is_email_spoofable",
            description=(
                "Composite indicator representing potential email spoofability "
                "based on SPF and DMARC configuration."
            ),
            feature_type="boolean",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            security_relevance=(
                "Weak email authentication may increase impersonation risk, "
                "but does not independently indicate website phishing."
            ),
        ),
    }

    # =========================================================================
    # 3. BACKWARD-COMPATIBILITY ALIASES
    # =========================================================================

    """
    Older DNS feature names can appear in experimental extractors or datasets.

    The canonical names above MUST be used for ML training and inference.

    Aliases allow the schema to safely accept older raw dictionaries without
    changing the model feature order.
    """

    FEATURE_ALIASES: Dict[str, str] = {

        # Older A / AAAA names
        "has_a_record": "has_a_records",
        "has_aaaa_record": "has_aaaa_records",

        # Older CNAME naming
        "has_cname": "has_cname_record",

        # Older nameserver naming
        "ns_record_count": "ns_count",

        # Older MX naming
        "has_mx_record": "has_mx_records",

        # Older TXT naming
        "has_txt_record": "has_txt_records",

        # Older SPF naming
        "has_spf": "has_spf_record",

        # Older DMARC naming
        "has_dmarc": "has_dmarc_record",

        # Initial/simple DNS schema compatibility
        "private_ip_detected": "has_routing_anomalies",

        # Initial simple TTL field
        "dns_ttl": "avg_ttl_value",

        # Initial simple CNAME field
        "cname_record_count": "cname_count",
    }

    # =========================================================================
    # 4. CATEGORICAL POLICY MAPPINGS
    # =========================================================================

    SPF_STRICTNESS_MAP: Dict[str, float] = {
        "invalid": 0.0,
        "none": 0.0,
        "unknown": 0.0,
        "dangerously_permissive": 1.0,
        "dangerous": 1.0,
        "permissive": 2.0,
        "neutral": 3.0,
        "strict": 4.0,
    }

    DMARC_POLICY_MAP: Dict[str, float] = {
        "invalid": 0.0,
        "unknown": 0.0,
        "none": 1.0,
        "quarantine": 2.0,
        "reject": 3.0,
    }

    # =========================================================================
    # 5. BOOLEAN STRING REPRESENTATIONS
    # =========================================================================

    TRUE_VALUES = {
        "true",
        "1",
        "yes",
        "y",
        "on",
        "enabled",
        "enable",
        "present",
        "detected",
        "valid",
    }

    FALSE_VALUES = {
        "false",
        "0",
        "no",
        "n",
        "off",
        "disabled",
        "disable",
        "absent",
        "not_detected",
        "invalid",
        "none",
    }

    # =========================================================================
    # 6. SCHEMA INFORMATION
    # =========================================================================

    @classmethod
    def get_schema(cls) -> List[str]:
        """
        Return a copy of the canonical feature order.

        Returns
        -------
        List[str]
            Ordered list of ML features.
        """

        return list(cls.EXPECTED_FEATURE_ORDER)

    # =========================================================================
    # 7. FEATURE COUNT
    # =========================================================================

    @classmethod
    def feature_count(cls) -> int:
        """
        Return the number of canonical ML features.
        """

        return len(cls.EXPECTED_FEATURE_ORDER)

    # =========================================================================
    # 8. FEATURE EXISTENCE CHECK
    # =========================================================================

    @classmethod
    def has_feature(cls, feature_name: str) -> bool:
        """
        Check whether a feature exists in the canonical schema.
        """

        return feature_name in cls.EXPECTED_FEATURE_ORDER

    # =========================================================================
    # 9. FEATURE METADATA ACCESS
    # =========================================================================

    @classmethod
    def get_feature_metadata(
        cls,
        feature_name: str,
    ) -> Optional[FeatureInfo]:
        """
        Return metadata for a feature.
        """

        return cls.FEATURE_METADATA.get(feature_name)

    # =========================================================================
    # 10. FEATURE DESCRIPTION
    # =========================================================================

    @classmethod
    def describe_feature(
        cls,
        feature_name: str,
    ) -> str:
        """
        Return a human-readable feature description.

        Unknown features return a safe fallback message.
        """

        metadata = cls.FEATURE_METADATA.get(feature_name)

        if metadata is None:
            return f"No metadata available for feature '{feature_name}'."

        return metadata.description

    # =========================================================================
    # 11. FEATURE ALIAS NORMALIZATION
    # =========================================================================

    @classmethod
    def normalize_feature_names(
        cls,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert legacy feature names into canonical feature names.

        Canonical values always take precedence over aliases.

        Example
        -------
        Input:

            {
                "has_a_record": True,
                "has_mx_record": True
            }

        Output:

            {
                "has_a_records": True,
                "has_mx_records": True
            }
        """

        normalized: Dict[str, Any] = {}

        # First copy canonical values.
        for key, value in raw_features.items():
            if key in cls.EXPECTED_FEATURE_ORDER:
                normalized[key] = value

        # Then use aliases only when canonical value is missing.
        for alias, canonical_name in cls.FEATURE_ALIASES.items():

            if alias in raw_features and canonical_name not in normalized:
                normalized[canonical_name] = raw_features[alias]

                logger.debug(
                    "DNS feature alias '%s' normalized to '%s'.",
                    alias,
                    canonical_name,
                )

        return normalized

    # =========================================================================
    # 12. BOOLEAN NORMALIZATION
    # =========================================================================

    @classmethod
    def normalize_boolean(
        cls,
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert a variety of boolean representations into 0.0 or 1.0.

        Supported examples:

            True
            False
            1
            0
            "true"
            "false"
            "yes"
            "no"
            "enabled"
            "disabled"

        Invalid values fall back to `default`.
        """

        if value is None:
            return float(default)

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):

            if not math.isfinite(float(value)):
                return float(default)

            return 1.0 if float(value) != 0.0 else 0.0

        if isinstance(value, str):

            normalized = value.strip().lower()

            if normalized in cls.TRUE_VALUES:
                return 1.0

            if normalized in cls.FALSE_VALUES:
                return 0.0

        logger.warning(
            "Unable to normalize boolean value '%r'. Using default=%s.",
            value,
            default,
        )

        return float(default)

    # =========================================================================
    # 13. NUMERIC NORMALIZATION
    # =========================================================================

    @classmethod
    def normalize_numeric(
        cls,
        value: Any,
        default: float = 0.0,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        """
        Convert an arbitrary value into a safe finite float.

        Invalid values become the supplied default.

        Values can optionally be constrained using minimum and maximum.
        """

        if value is None:
            return float(default)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Unable to convert DNS feature value '%r' to float. "
                "Using default=%s.",
                value,
                default,
            )
            return float(default)

        if not math.isfinite(numeric_value):
            logger.warning(
                "Non-finite DNS feature value '%r' detected. "
                "Using default=%s.",
                value,
                default,
            )
            return float(default)

        if minimum is not None and numeric_value < minimum:
            logger.warning(
                "DNS feature value %.4f is below minimum %.4f. "
                "Clamping value.",
                numeric_value,
                minimum,
            )
            numeric_value = minimum

        if maximum is not None and numeric_value > maximum:
            logger.warning(
                "DNS feature value %.4f is above maximum %.4f. "
                "Clamping value.",
                numeric_value,
                maximum,
            )
            numeric_value = maximum

        return float(numeric_value)

    # =========================================================================
    # 14. SPF POLICY NORMALIZATION
    # =========================================================================

    @classmethod
    def normalize_spf_policy(
        cls,
        value: Any,
    ) -> float:
        """
        Convert SPF policy labels into an ordinal numeric score.

        Mapping:

            invalid / none             -> 0
            dangerously_permissive    -> 1
            permissive                 -> 2
            neutral                    -> 3
            strict                     -> 4
        """

        if value is None:
            return 0.0

        if isinstance(value, (int, float)):

            return cls.normalize_numeric(
                value=value,
                default=0.0,
                minimum=0.0,
                maximum=4.0,
            )

        normalized = str(value).strip().lower()

        # Accept common SPF mechanisms directly.
        if normalized in {"+all", "all"}:
            return 1.0

        if normalized == "?all":
            return 2.0

        if normalized == "~all":
            return 3.0

        if normalized == "-all":
            return 4.0

        return cls.SPF_STRICTNESS_MAP.get(normalized, 0.0)

    # =========================================================================
    # 15. DMARC POLICY NORMALIZATION
    # =========================================================================

    @classmethod
    def normalize_dmarc_policy(
        cls,
        value: Any,
    ) -> float:
        """
        Convert DMARC policy labels into an ordinal score.

        Mapping:

            unknown / invalid -> 0
            none              -> 1
            quarantine        -> 2
            reject            -> 3
        """

        if value is None:
            return 0.0

        if isinstance(value, (int, float)):

            return cls.normalize_numeric(
                value=value,
                default=0.0,
                minimum=0.0,
                maximum=3.0,
            )

        normalized = str(value).strip().lower()

        return cls.DMARC_POLICY_MAP.get(
            normalized,
            0.0,
        )

    # =========================================================================
    # 16. RAW FEATURE EXTRACTION
    # =========================================================================

    @classmethod
    def normalize_raw_features(
        cls,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Normalize raw DNS feature telemetry into a canonical numeric dictionary.

        Parameters
        ----------
        raw_features:
            Raw feature dictionary generated by the DNS feature extractor.

        Returns
        -------
        Dict[str, float]
            Canonical feature dictionary containing every expected feature.
        """

        if raw_features is None:
            raw_features = {}

        if not isinstance(raw_features, Mapping):
            raise TypeError(
                "raw_features must be a mapping/dictionary."
            )

        # Resolve aliases first.
        canonical_raw = cls.normalize_feature_names(
            raw_features
        )

        normalized: Dict[str, float] = {}

        for feature_name in cls.EXPECTED_FEATURE_ORDER:

            metadata = cls.FEATURE_METADATA.get(feature_name)

            if metadata is None:
                logger.warning(
                    "Feature '%s' does not have metadata. "
                    "Using generic numeric normalization.",
                    feature_name,
                )

                metadata = FeatureInfo(
                    name=feature_name,
                    description="No metadata available.",
                    feature_type="numeric",
                )

            raw_value = canonical_raw.get(
                feature_name,
                metadata.default,
            )

            # -----------------------------------------------------------------
            # Boolean
            # -----------------------------------------------------------------

            if metadata.feature_type == "boolean":

                normalized_value = cls.normalize_boolean(
                    raw_value,
                    default=metadata.default,
                )

            # -----------------------------------------------------------------
            # Ordinal
            # -----------------------------------------------------------------

            elif metadata.feature_type == "ordinal":

                if feature_name == "spf_strictness_score":

                    # Prefer the already computed score.
                    if feature_name in canonical_raw:
                        normalized_value = cls.normalize_numeric(
                            raw_value,
                            default=metadata.default,
                            minimum=metadata.min_value,
                            maximum=metadata.max_value,
                        )
                    else:
                        normalized_value = cls.normalize_spf_policy(
                            canonical_raw.get(
                                "spf_strictness",
                                "none",
                            )
                        )

                elif feature_name in {
                    "dmarc_policy_score",
                    "dmarc_subdomain_policy_score",
                }:

                    if feature_name in canonical_raw:

                        normalized_value = cls.normalize_numeric(
                            raw_value,
                            default=metadata.default,
                            minimum=metadata.min_value,
                            maximum=metadata.max_value,
                        )

                    elif feature_name == "dmarc_policy_score":

                        normalized_value = cls.normalize_dmarc_policy(
                            canonical_raw.get(
                                "dmarc_policy",
                                "unknown",
                            )
                        )

                    else:

                        normalized_value = cls.normalize_dmarc_policy(
                            canonical_raw.get(
                                "dmarc_subdomain_policy",
                                "unknown",
                            )
                        )

                else:

                    normalized_value = cls.normalize_numeric(
                        raw_value,
                        default=metadata.default,
                        minimum=metadata.min_value,
                        maximum=metadata.max_value,
                    )

            # -----------------------------------------------------------------
            # Count / Numeric
            # -----------------------------------------------------------------

            else:

                normalized_value = cls.normalize_numeric(
                    raw_value,
                    default=metadata.default,
                    minimum=metadata.min_value,
                    maximum=metadata.max_value,
                )

            normalized[feature_name] = normalized_value

        return normalized

    # =========================================================================
    # 17. ALIGN AND VALIDATE
    # =========================================================================

    @classmethod
    def align_and_validate(
        cls,
        raw_features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        Normalize, validate and align raw DNS features into a one-row DataFrame.

        This is the primary function that predictor.py should use before
        sending data to the ML model.

        Returns
        -------
        pandas.DataFrame
            Exactly one row and exactly 37 columns in canonical order.
        """

        normalized_features = cls.normalize_raw_features(
            raw_features
        )

        # Construct DataFrame in EXACT canonical order.
        dataframe = pd.DataFrame(
            [
                normalized_features
            ],
            columns=cls.EXPECTED_FEATURE_ORDER,
        )

        # Force numeric dtype.
        dataframe = dataframe.astype(float)

        # Final NaN protection.
        dataframe = dataframe.fillna(0.0)

        # Final infinity protection.
        dataframe = dataframe.replace(
            [float("inf"), float("-inf")],
            0.0,
        )

        # Final structural validation.
        cls.validate_dataframe(
            dataframe
        )

        return dataframe

    # =========================================================================
    # 18. DATAFRAME VALIDATION
    # =========================================================================

    @classmethod
    def validate_dataframe(
        cls,
        dataframe: pd.DataFrame,
    ) -> bool:
        """
        Validate a model-ready DNS feature DataFrame.

        Validation includes:

            - DataFrame type
            - exactly one row for inference
            - expected feature count
            - expected feature names
            - deterministic feature ordering
            - numeric values
            - no NaN
            - no infinity
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "DNS feature input must be a pandas DataFrame."
            )

        if len(dataframe.columns) != cls.feature_count():
            raise ValueError(
                "DNS feature count mismatch. "
                f"Expected {cls.feature_count()} features, "
                f"received {len(dataframe.columns)}."
            )

        actual_columns = list(dataframe.columns)

        if actual_columns != cls.EXPECTED_FEATURE_ORDER:

            missing = [
                feature
                for feature in cls.EXPECTED_FEATURE_ORDER
                if feature not in actual_columns
            ]

            unexpected = [
                feature
                for feature in actual_columns
                if feature not in cls.EXPECTED_FEATURE_ORDER
            ]

            raise ValueError(
                "DNS feature schema mismatch.\n"
                f"Missing features: {missing}\n"
                f"Unexpected features: {unexpected}\n"
                f"Expected order: {cls.EXPECTED_FEATURE_ORDER}\n"
                f"Received order: {actual_columns}"
            )

        if not all(
            pd.api.types.is_numeric_dtype(dataframe[column])
            for column in dataframe.columns
        ):
            raise TypeError(
                "DNS feature DataFrame contains non-numeric columns."
            )

        if dataframe.isna().any().any():

            raise ValueError(
                "DNS feature DataFrame contains NaN values."
            )

        if dataframe.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        ).isna().any().any():

            raise ValueError(
                "DNS feature DataFrame contains infinite values."
            )

        return True

    # =========================================================================
    # 19. RAW FEATURE VALIDATION REPORT
    # =========================================================================

    @classmethod
    def validation_report(
        cls,
        raw_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a non-destructive validation report.

        Unlike align_and_validate(), this function does not raise an error
        for normal missing values.

        Useful for debugging feature extraction.
        """

        if raw_features is None:
            raw_features = {}

        if not isinstance(raw_features, Mapping):

            return {
                "valid": False,
                "schema_version": SCHEMA_VERSION,
                "error": "raw_features must be a mapping.",
            }

        normalized_names = cls.normalize_feature_names(
            raw_features
        )

        expected = set(cls.EXPECTED_FEATURE_ORDER)
        received = set(normalized_names.keys())

        missing = sorted(
            expected - received
        )

        extra = sorted(
            received - expected
        )

        invalid_types: List[str] = []

        for feature_name in cls.EXPECTED_FEATURE_ORDER:

            if feature_name not in normalized_names:
                continue

            value = normalized_names[feature_name]

            metadata = cls.FEATURE_METADATA.get(
                feature_name
            )

            if metadata is None:
                continue

            if metadata.feature_type == "boolean":

                if isinstance(value, str):
                    if (
                        value.strip().lower()
                        not in cls.TRUE_VALUES
                        and value.strip().lower()
                        not in cls.FALSE_VALUES
                    ):
                        invalid_types.append(feature_name)

                elif not isinstance(
                    value,
                    (bool, int, float),
                ):
                    invalid_types.append(feature_name)

            else:

                try:
                    numeric_value = float(value)

                    if not math.isfinite(numeric_value):
                        invalid_types.append(feature_name)

                except (TypeError, ValueError):

                    invalid_types.append(feature_name)

        return {
            "valid": len(invalid_types) == 0,
            "schema_version": SCHEMA_VERSION,
            "expected_feature_count": cls.feature_count(),
            "received_feature_count": len(received),
            "missing_features": missing,
            "extra_features": extra,
            "invalid_features": sorted(
                invalid_types
            ),
        }

    # =========================================================================
    # 20. STRICT VALIDATION
    # =========================================================================

    @classmethod
    def validate_raw_features(
        cls,
        raw_features: Mapping[str, Any],
        strict: bool = False,
    ) -> bool:
        """
        Validate raw DNS feature telemetry.

        Parameters
        ----------
        raw_features:
            Raw feature dictionary.

        strict:
            If True, missing or extra features cause failure.
            If False, missing features are allowed because the schema
            provides safe defaults.

        Returns
        -------
        bool
            True when validation succeeds.
        """

        report = cls.validation_report(
            raw_features
        )

        if report["invalid_features"]:
            raise ValueError(
                "Invalid DNS features detected: "
                f"{report['invalid_features']}"
            )

        if strict and report["missing_features"]:
            raise ValueError(
                "Missing required DNS features: "
                f"{report['missing_features']}"
            )

        if strict and report["extra_features"]:
            raise ValueError(
                "Unexpected DNS features detected: "
                f"{report['extra_features']}"
            )

        return True

    # =========================================================================
    # 21. MODEL FEATURE COMPATIBILITY
    # =========================================================================

    @classmethod
    def validate_model_features(
        cls,
        model_features: List[str],
    ) -> bool:
        """
        Validate that the ML model was trained using the same feature order.

        This is especially important when using:

            RandomForest
            XGBoost
            LightGBM
            Logistic Regression
            SVM
            Neural Networks

        A model can produce completely incorrect results if feature order
        changes while column names remain unavailable to the model.
        """

        expected = cls.EXPECTED_FEATURE_ORDER

        if list(model_features) != expected:

            raise ValueError(
                "ML model feature schema does not match DNS schema.\n"
                f"Expected: {expected}\n"
                f"Received: {list(model_features)}"
            )

        return True

    # =========================================================================
    # 22. EMPTY FEATURE TEMPLATE
    # =========================================================================

    @classmethod
    def empty_feature_dict(cls) -> Dict[str, float]:
        """
        Return a complete zero-valued DNS feature dictionary.

        Useful for initializing feature extraction.
        """

        return {
            feature_name: 0.0
            for feature_name in cls.EXPECTED_FEATURE_ORDER
        }

    # =========================================================================
    # 23. FEATURE VECTOR
    # =========================================================================

    @classmethod
    def to_feature_vector(
        cls,
        raw_features: Mapping[str, Any],
    ) -> List[float]:
        """
        Convert raw DNS features into an ordered ML feature vector.

        Returns
        -------
        List[float]
            Ordered numeric feature vector.
        """

        dataframe = cls.align_and_validate(
            raw_features
        )

        return dataframe.iloc[0].tolist()

    # =========================================================================
    # 24. FEATURE DICTIONARY FROM DATAFRAME
    # =========================================================================

    @classmethod
    def dataframe_to_dict(
        cls,
        dataframe: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Convert a model-ready DataFrame back into an ordered dictionary.
        """

        cls.validate_dataframe(
            dataframe
        )

        if len(dataframe) != 1:
            raise ValueError(
                "dataframe_to_dict() expects exactly one row."
            )

        return {
            feature_name: float(
                dataframe.iloc[0][feature_name]
            )
            for feature_name in cls.EXPECTED_FEATURE_ORDER
        }

    # =========================================================================
    # 25. FEATURE METADATA DICTIONARY
    # =========================================================================

    @classmethod
    def metadata_dict(
        cls,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return feature metadata in serializable dictionary form.

        Useful for:

            - API responses
            - XAI
            - debugging
            - reports
            - documentation
        """

        result: Dict[str, Dict[str, Any]] = {}

        for feature_name in cls.EXPECTED_FEATURE_ORDER:

            metadata = cls.FEATURE_METADATA.get(
                feature_name
            )

            if metadata is None:
                continue

            result[feature_name] = {
                "description": metadata.description,
                "feature_type": metadata.feature_type,
                "default": metadata.default,
                "min_value": metadata.min_value,
                "max_value": metadata.max_value,
                "security_relevance": metadata.security_relevance,
            }

        return result

    # =========================================================================
    # 26. SCHEMA SUMMARY
    # =========================================================================

    @classmethod
    def schema_summary(cls) -> Dict[str, Any]:
        """
        Return a compact schema summary.
        """

        feature_types: Dict[str, int] = {}

        for feature_name in cls.EXPECTED_FEATURE_ORDER:

            metadata = cls.FEATURE_METADATA.get(
                feature_name
            )

            if metadata is None:
                continue

            feature_type = metadata.feature_type

            feature_types[feature_type] = (
                feature_types.get(feature_type, 0) + 1
            )

        return {
            "agent": "DNS Security AI Agent",
            "schema_version": SCHEMA_VERSION,
            "feature_count": cls.feature_count(),
            "feature_types": feature_types,
            "features": cls.get_schema(),
        }


# =============================================================================
# MODULE-LEVEL HELPER FUNCTIONS
# =============================================================================

def normalize_dns_features(
    raw_features: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Convenience wrapper around DNSFeatureSchema.align_and_validate().

    Example
    -------
        dataframe = normalize_dns_features(raw_features)
    """

    return DNSFeatureSchema.align_and_validate(
        raw_features
    )


def get_dns_feature_names() -> List[str]:
    """
    Return the canonical DNS feature names.
    """

    return DNSFeatureSchema.get_schema()


def get_dns_feature_count() -> int:
    """
    Return the canonical DNS feature count.
    """

    return DNSFeatureSchema.feature_count()


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # -------------------------------------------------------------------------
    # Example raw DNS telemetry
    # -------------------------------------------------------------------------

    example_features = {

        # Routing
        "has_a_records": True,
        "has_aaaa_records": False,
        "has_cname_record": True,
        "a_record_count": 2,
        "aaaa_record_count": 0,
        "ns_count": 2,
        "cname_count": 1,
        "total_resolved_ips": 2,
        "has_routing_anomalies": False,

        # TTL / Fast Flux
        "min_ttl_value": 300,
        "max_ttl_value": 3600,
        "avg_ttl_value": 1800,
        "is_fast_flux_candidate": False,
        "is_active_fast_flux": False,

        # MX
        "has_mx_records": True,
        "mx_record_count": 2,
        "uses_free_mail_provider": False,
        "uses_disposable_mail_provider": False,
        "has_suspicious_mx_exchange": False,
        "lowest_mx_preference": 10,

        # TXT
        "has_txt_records": True,
        "txt_record_count": 4,
        "has_domain_verification": True,
        "has_suspicious_long_txt": False,
        "has_base64_payload_in_txt": False,
        "avg_txt_length": 80,

        # SPF
        "has_spf_record": True,
        "spf_record_count": 1,
        "has_multiple_spf_records": False,
        "spf_includes_count": 2,
        "spf_strictness": "strict",

        # DMARC
        "has_dmarc_record": True,
        "dmarc_policy": "reject",
        "dmarc_subdomain_policy": "reject",

        # Composite
        "is_email_spoofable": False,
    }

    # -------------------------------------------------------------------------
    # Validation report
    # -------------------------------------------------------------------------

    report = DNSFeatureSchema.validation_report(
        example_features
    )

    logger.info(
        "Validation report: %s",
        report,
    )

    # -------------------------------------------------------------------------
    # Convert to DataFrame
    # -------------------------------------------------------------------------

    dataframe = DNSFeatureSchema.align_and_validate(
        example_features
    )

    logger.info(
        "DNS feature count: %d",
        len(dataframe.columns),
    )

    logger.info(
        "DNS feature DataFrame shape: %s",
        dataframe.shape,
    )

    logger.info(
        "DNS feature order: %s",
        list(dataframe.columns),
    )

    # -------------------------------------------------------------------------
    # Feature vector
    # -------------------------------------------------------------------------

    feature_vector = DNSFeatureSchema.to_feature_vector(
        example_features
    )

    logger.info(
        "Feature vector length: %d",
        len(feature_vector),
    )

    # -------------------------------------------------------------------------
    # Schema summary
    # -------------------------------------------------------------------------

    logger.info(
        "Schema summary: %s",
        DNSFeatureSchema.schema_summary(),
    )