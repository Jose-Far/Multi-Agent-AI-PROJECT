"""
Multi-Agent Cybersecurity Orchestrator
======================================

Coordinates the active cybersecurity agents:

    1. URL AI Agent
    2. HTML AI Agent
    3. SSL AI Agent
    4. DNS AI Agent
    5. Visual AI Agent
    6. Threat Intelligence AI Agent

Pipeline:

    Unified Feature Vector
            |
            +--> URL Agent
            +--> HTML Agent
            +--> SSL Agent
            +--> DNS Agent
            +--> Visual Agent
            +--> Threat Intelligence Agent
            |
            v
       Decision Fusion Engine
            |
            v
       Final Security Decision

Design principles:

- All six production agents are preserved.
- The complete unified vector is passed to every agent.
- The orchestrator does NOT reject an agent merely because its
  feature block is absent.
- Each agent remains responsible for its own feature extraction/
  validation requirements.
- DNS retains canonical 35-feature validation when a DNS feature
  block is supplied.
- Agent failures are isolated.
- Missing signals are never converted into legitimate signals.
- Only usable agent signals are forwarded to Fusion.
- Threat Intelligence is preserved and forwarded to Fusion.
- Fusion v17.0.0 remains the decision layer.
- Existing public methods and output keys are preserved.
"""

from __future__ import annotations
import asyncio
from performance.config import PerformanceConfig

import asyncio
import inspect
import logging
import time

from typing import (
    Any,
    Dict,
    List,
    Optional,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT IMPORTS
# ============================================================================

from agents.url_agent.url_agent import URLAIAgent
from agents.html_agent.html_agent import HTMLAIAgent
from agents.ssl_agent.ssl_agent import SSLAIAgent
from agents.dns_agent.dns_agent import DNSAIAgent
from agents.dns_agent.feature_schema import DNSFeatureSchema
from agents.visual_agent.visual_agent import VisualAIAgent
from agents.threat_agent.threat_agent import (
    ThreatIntelligenceAIAgent,
)


# ============================================================================
# FUSION IMPORT
# ============================================================================

from agents.fusion import (
    DecisionFusionEngine,
    ACTIVE_AGENTS,
    AGENT_FEATURE_MAPPING,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

ORCHESTRATOR_NAME = (
    "Multi-Agent Cybersecurity Orchestrator"
)

ORCHESTRATOR_VERSION = (
    "10.2.0"
)


# ============================================================================
# AGENT NAMES
# ============================================================================

URL_AGENT_NAME = "URL_AI_Agent"
HTML_AGENT_NAME = "HTML_AI_Agent"
SSL_AGENT_NAME = "SSL_AI_Agent"
DNS_AGENT_NAME = "DNS_AI_Agent"
VISUAL_AGENT_NAME = "Visual_AI_Agent"
THREAT_INTEL_AGENT_NAME = "Threat_Intel_Agent"


# ============================================================================
# ACTIVE PRODUCTION AGENTS
# ============================================================================

ACTIVE_AGENT_NAMES = [
    URL_AGENT_NAME,
    HTML_AGENT_NAME,
    SSL_AGENT_NAME,
    DNS_AGENT_NAME,
    VISUAL_AGENT_NAME,
    THREAT_INTEL_AGENT_NAME,
]


# ============================================================================
# FEATURE MAPPING
# ============================================================================

FEATURE_MAPPING = {

    URL_AGENT_NAME:
        "url_features",

    HTML_AGENT_NAME:
        "html_features",

    SSL_AGENT_NAME:
        "ssl_features",

    DNS_AGENT_NAME:
        "dns_features",

    VISUAL_AGENT_NAME:
        "visual_features",

    THREAT_INTEL_AGENT_NAME:
        "threat_features",
}


# ============================================================================
# KNOWN FEATURE COUNTS
# ============================================================================

DNS_FEATURE_COUNT = 35


# ============================================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================================

class MultiAgentOrchestrator:
    """
    Production orchestrator for the Multi-Agent AI Cybersecurity Analyst.

    Responsibilities:

    1. Initialize all active agents.
    2. Maintain the central agent registry.
    3. Dispatch agents independently.
    4. Pass the complete unified vector to each agent.
    5. Isolate individual agent failures.
    6. Normalize agent responses.
    7. Preserve unavailable signals.
    8. Validate DNS 35-feature structure when supplied.
    9. Prepare usable signals for Fusion.
    10. Execute Decision Fusion.
    11. Return a stable application-facing contract.
    """

    # ========================================================================
    # CONSTRUCTOR
    # ========================================================================

    def __init__(
        self,
        url_agent: Optional[Any] = None,
        html_agent: Optional[Any] = None,
        ssl_agent: Optional[Any] = None,
        dns_agent: Optional[Any] = None,
        visual_agent: Optional[Any] = None,
        threat_intel_agent: Optional[Any] = None,
        fusion_engine: Optional[
            DecisionFusionEngine
        ] = None,
    ) -> None:

        logger.info(
            "Initializing %s v%s...",
            ORCHESTRATOR_NAME,
            ORCHESTRATOR_VERSION,
        )

        self.orchestrator_name = (
            ORCHESTRATOR_NAME
        )

        self.orchestrator_version = (
            ORCHESTRATOR_VERSION
        )

        # ====================================================================
        # AGENTS
        # ====================================================================

        self.url_agent = (
            url_agent
            if url_agent is not None
            else URLAIAgent()
        )

        self.html_agent = (
            html_agent
            if html_agent is not None
            else HTMLAIAgent()
        )

        self.ssl_agent = (
            ssl_agent
            if ssl_agent is not None
            else SSLAIAgent()
        )

        self.dns_agent = (
            dns_agent
            if dns_agent is not None
            else DNSAIAgent()
        )

        self.visual_agent = (
            visual_agent
            if visual_agent is not None
            else VisualAIAgent()
        )

        self.threat_intel_agent = (
            threat_intel_agent
            if threat_intel_agent is not None
            else ThreatIntelligenceAIAgent()
        )

        # ====================================================================
        # FUSION
        # ====================================================================

        self.fusion_engine = (
            fusion_engine
            if fusion_engine is not None
            else DecisionFusionEngine()
        )

        # ====================================================================
        # CENTRAL AGENT REGISTRY
        # ====================================================================

        self.agents = {

            URL_AGENT_NAME: {
                "instance":
                    self.url_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        URL_AGENT_NAME
                    ],
            },

            HTML_AGENT_NAME: {
                "instance":
                    self.html_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        HTML_AGENT_NAME
                    ],
            },

            SSL_AGENT_NAME: {
                "instance":
                    self.ssl_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        SSL_AGENT_NAME
                    ],
            },

            DNS_AGENT_NAME: {
                "instance":
                    self.dns_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        DNS_AGENT_NAME
                    ],
            },

            VISUAL_AGENT_NAME: {
                "instance":
                    self.visual_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        VISUAL_AGENT_NAME
                    ],
            },

            THREAT_INTEL_AGENT_NAME: {
                "instance":
                    self.threat_intel_agent,

                "method":
                    "analyze",

                "feature_key":
                    FEATURE_MAPPING[
                        THREAT_INTEL_AGENT_NAME
                    ],
            },
        }

        # ====================================================================
        # COMPATIBILITY ALIAS
        # ====================================================================

        self.agent_registry = (
            self.agents
        )

        # ====================================================================
        # CONFIGURATION VALIDATION
        # ====================================================================

        self._validate_agent_configuration()

        logger.info(
            "Multi-Agent Orchestrator initialized successfully."
        )

        logger.info(
            "Active agents: %s",
            sorted(
                ACTIVE_AGENT_NAMES
            ),
        )


    # ========================================================================
    # CONFIGURATION VALIDATION
    # ========================================================================

    def _validate_agent_configuration(
        self,
    ) -> None:
        """
        Validate the active Orchestrator/Fusion configuration.

        Fusion may contain mappings for future/inactive agents.
        Only the six active production agents are validated here.
        """

        expected_agents = set(
            ACTIVE_AGENT_NAMES
        )

        # --------------------------------------------------------------------
        # Orchestrator registry
        # --------------------------------------------------------------------

        registered_agents = set(
            self.agents.keys()
        )

        if registered_agents != expected_agents:

            raise RuntimeError(
                "Orchestrator agent registry mismatch.\n"
                f"Expected: {sorted(expected_agents)}\n"
                f"Found: {sorted(registered_agents)}"
            )

        # --------------------------------------------------------------------
        # Fusion active agents
        # --------------------------------------------------------------------

        fusion_active_agents = set(
            ACTIVE_AGENTS
        )

        if fusion_active_agents != expected_agents:

            raise RuntimeError(
                "Orchestrator and Fusion active-agent "
                "configuration mismatch.\n"
                f"Orchestrator: {sorted(expected_agents)}\n"
                f"Fusion: {sorted(fusion_active_agents)}"
            )

        # --------------------------------------------------------------------
        # Feature mapping consistency
        # --------------------------------------------------------------------

        mapping_errors = []

        for agent_name in ACTIVE_AGENT_NAMES:

            orchestrator_feature = (
                FEATURE_MAPPING.get(
                    agent_name
                )
            )

            fusion_feature = (
                AGENT_FEATURE_MAPPING.get(
                    agent_name
                )
            )

            if (
                orchestrator_feature
                != fusion_feature
            ):

                mapping_errors.append(
                    {
                        "agent":
                            agent_name,

                        "orchestrator":
                            orchestrator_feature,

                        "fusion":
                            fusion_feature,
                    }
                )

        if mapping_errors:

            raise RuntimeError(
                "Orchestrator and Fusion feature mapping "
                "configuration mismatch.\n"
                f"Errors: {mapping_errors}"
            )

        # --------------------------------------------------------------------
        # Missing active mappings
        # --------------------------------------------------------------------

        missing_mappings = [

            agent_name

            for agent_name
            in ACTIVE_AGENT_NAMES

            if not FEATURE_MAPPING.get(
                agent_name
            )
        ]

        if missing_mappings:

            raise RuntimeError(
                "Active agents missing feature mappings: "
                f"{missing_mappings}"
            )

        logger.info(
            "Orchestrator/Fusion configuration validation passed."
        )


    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return stable public orchestrator status.
        """

        fusion_status = {}

        try:

            if hasattr(
                self.fusion_engine,
                "get_status",
            ):

                fusion_status = (
                    self.fusion_engine.get_status()
                )

        except Exception as exc:

            logger.warning(
                "Unable to retrieve Fusion status: %s",
                exc,
            )

        return {

            "status":
                "ready",

            "orchestrator_name":
                self.orchestrator_name,

            "orchestrator_version":
                self.orchestrator_version,

            "registered_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "agent_count":
                len(
                    ACTIVE_AGENT_NAMES
                ),

            "active_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "active_agent_count":
                len(
                    ACTIVE_AGENT_NAMES
                ),

            "agent_feature_mapping":
                dict(
                    FEATURE_MAPPING
                ),

            "fusion_active_agents":
                sorted(
                    ACTIVE_AGENTS
                ),

            "fusion_feature_mapping":
                dict(
                    AGENT_FEATURE_MAPPING
                ),

            "configuration_consistent":
                True,

            "url_agent":
                {
                    "enabled":
                        True,

                    "feature_key":
                        "url_features",
                },

            "html_agent":
                {
                    "enabled":
                        True,

                    "feature_key":
                        "html_features",

                    "feature_count":
                        50,
                },

            "ssl_agent":
                {
                    "enabled":
                        True,

                    "feature_key":
                        "ssl_features",
                },

            "dns_agent":
                {
                    "enabled":
                        True,

                    "feature_key":
                        "dns_features",

                    "feature_count":
                        DNS_FEATURE_COUNT,

                    "schema_version":
                        "1.0.0",

                    "model_type":
                        "XGBoost",

                    "model_path":
                        "data/models/dns_agent/"
                        "dns_xgb_model.json",

                    "label_definition":
                        {
                            "0":
                                "legitimate",

                            "1":
                                "phishing",
                        },
                },

            "visual_agent":
                {
                    "enabled":
                        True,

                    "feature_key":
                        "visual_features",
                },

            "threat_intelligence_agent":
                {
                    "enabled":
                        True,

                    "conditional":
                        False,

                    "feature_key":
                        "threat_features",

                    "agent_name":
                        THREAT_INTEL_AGENT_NAME,
                },

            "future_fusion_agents":
                sorted(
                    [
                        agent
                        for agent
                        in AGENT_FEATURE_MAPPING
                        if agent
                        not in ACTIVE_AGENT_NAMES
                    ]
                ),

            "fusion_engine":
                fusion_status,
        }


    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Return lightweight orchestrator health information.
        """

        configuration_consistent = True

        try:

            self._validate_agent_configuration()

        except Exception:

            configuration_consistent = False

        fusion_ready = False

        try:

            fusion_health = getattr(
                self.fusion_engine,
                "health_check",
                None,
            )

            if callable(
                fusion_health
            ):

                health = fusion_health()

                if isinstance(
                    health,
                    dict,
                ):

                    fusion_ready = bool(
                        health.get(
                            "healthy",
                            health.get(
                                "status"
                            )
                            == "ready",
                        )
                    )

                else:

                    fusion_ready = bool(
                        health
                    )

            else:

                fusion_ready = True

        except Exception as exc:

            logger.warning(
                "Fusion health check failed: %s",
                exc,
            )

            fusion_ready = False

        return {

            "healthy":
                bool(
                    configuration_consistent
                    and fusion_ready
                ),

            "orchestrator":
                self.orchestrator_name,

            "version":
                self.orchestrator_version,

            "active_agent_count":
                len(
                    ACTIVE_AGENT_NAMES
                ),

            "fusion_ready":
                fusion_ready,

            "configuration_consistent":
                configuration_consistent,
        }


    # ========================================================================
    # INPUT ERROR RESULT
    # ========================================================================

    def _build_input_error_result(
        self,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Stable result for invalid orchestrator input.
        """

        return {

            "orchestrator":
                self.orchestrator_name,

            "orchestrator_version":
                self.orchestrator_version,

            "analysis_status":
                "error",

            "signal_available":
                False,

            "execution_time_ms":
                0.0,

            "active_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "agent_count":
                len(
                    ACTIVE_AGENT_NAMES
                ),

            "successful_agents":
                [],

            "failed_agents":
                [],

            "unavailable_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "usable_agents":
                [],

            "usable_agent_count":
                0,

            "excluded_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "html_validation":
                {
                    "present":
                        False,

                    "analysis_status":
                        "unavailable",

                    "signal_available":
                        False,

                    "reason":
                        reason,
                },

            "ssl_validation":
                {
                    "present":
                        False,

                    "analysis_status":
                        "unavailable",

                    "signal_available":
                        False,

                    "reason":
                        reason,
                },

            "dns_validation":
                {
                    "present":
                        False,

                    "valid":
                        False,

                    "reason":
                        reason,
                },

            "agent_results":
                [],

            "agents":
                [],

            "fusion_signals":
                {},

            "threat_intelligence":
                None,

            "fusion":
                {},

            "fusion_result":
                {},

            "final_verdict":
                "unknown",

            "verdict":
                "unknown",

            "risk_score":
                None,

            "final_risk_score":
                None,

            "risk_level":
                "unknown",

            "confidence":
                0.0,

            "phishing_probability":
                None,

            "legitimate_probability":
                None,

            "consensus":
                {},

            "consensus_available":
                False,

            "decision_basis":
                {
                    "type":
                        "invalid_input",

                    "critical_override":
                        False,

                    "reasons":
                        [
                            reason
                        ],
                },

            # ----------------------------------------------------------------
            # Conflict & evidence state (Required top-level contract)
            # ----------------------------------------------------------------

            "conflict_detected":
                False,

            "evidence_state":
                "unavailable",

            "critical_evidence_override":
                False,

            "weighting":
                {},

            "reason":
                reason,

            "agent_status_consistency":
                {
                    "successful_agent_count":
                        0,

                    "failed_agent_count":
                        0,

                    "unavailable_agent_count":
                        len(
                            ACTIVE_AGENT_NAMES
                        ),

                    "fusion_usable_agent_count":
                        0,

                    "consistent":
                        True,
                },
        }


    # ========================================================================
    # UNAVAILABLE RESULT
    # ========================================================================

    def _build_unavailable_result(
        self,
        agent_name: str,
        feature_key: str,
        reason: str,
        started: float,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """
        Build a stable unavailable/error agent result.
        """

        result = {

            "schema_version":
                "1.0.0",

            "agent":
                agent_name,

            "agent_name":
                agent_name,

            "status":
                "error"
                if error is not None
                else "unavailable",

            "analysis_status":
                "error"
                if error is not None
                else "unavailable",

            "signal_available":
                False,

            "prediction":
                "unknown",

            "probability":
                None,

            "confidence":
                0.0,

            "risk_score":
                None,

            "risk_level":
                "unknown",

            "prediction_source":
                "unavailable",

            "model_status":
                "unavailable",

            "risk_factors":
                [],

            "evidence":
                [],

            "explanation":
                {
                    "summary":
                        (
                            f"{agent_name} did not produce "
                            "a usable security signal."
                        ),

                    "method":
                        "none",

                    "top_factors":
                        [],

                    "risk_factors":
                        [],

                    "protective_factors":
                        [],

                },

            "feature_key":
                feature_key,

            "reason":
                reason,

            "execution_time_ms":
                round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000.0,
                    3,
                ),
        }

        if error is not None:

            result["error"] = str(
                error
            )

            result["error_type"] = (
                type(error).__name__
            )

        else:

            result["error"] = None
            result["error_type"] = None

        result["execution_time_ms"] = int((time.perf_counter() - started) * 1000)
        
        from performance.metrics import PerformanceMetric
        metric = PerformanceMetric("analysis", agent_name)
        metric.start_time = started
        metric.stop(result.get("status", "success"))

        return result


    # ========================================================================
    # DNS VALIDATION
    # ========================================================================

    def _validate_dns_feature_block(
        self,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate DNS feature block when one is supplied.

        Important:
        Missing DNS features are NOT converted into legitimate evidence.

        If no DNS feature block is supplied, the agent still receives the
        complete unified vector and remains responsible for its own handling.
        """

        dns_features = unified_vector.get(
            "dns_features"
        )

        # --------------------------------------------------------------------
        # Missing block
        #
        # Do NOT reject here.
        # The DNS agent gets the complete unified vector.
        # --------------------------------------------------------------------

        if dns_features is None:

            return {

                "present":
                    False,

                "valid":
                    None,

                "feature_count":
                    0,

                "expected_feature_count":
                    DNS_FEATURE_COUNT,

                "reason":
                    (
                        "DNS feature block was not supplied. "
                        "DNS agent will receive the complete unified "
                        "vector and perform its own input handling."
                    ),
            }

        # --------------------------------------------------------------------
        # Wrong type
        # --------------------------------------------------------------------

        if not isinstance(
            dns_features,
            dict,
        ):

            return {

                "present":
                    True,

                "valid":
                    False,

                "feature_count":
                    0,

                "expected_feature_count":
                    DNS_FEATURE_COUNT,

                "reason":
                    (
                        "DNS feature block is present but is not a "
                        "dictionary."
                    ),
            }

        # --------------------------------------------------------------------
        # Empty block
        # --------------------------------------------------------------------

        if not dns_features:

            return {

                "present":
                    True,

                "valid":
                    False,

                "feature_count":
                    0,

                "expected_feature_count":
                    DNS_FEATURE_COUNT,

                "reason":
                    (
                        "DNS feature block is present but empty."
                    ),
            }

        # --------------------------------------------------------------------
        # Canonical schema validation
        # --------------------------------------------------------------------

        try:

            validation = (
                DNSFeatureSchema.validation_report(
                    dns_features
                )
            )

            if not isinstance(
                validation,
                dict,
            ):

                return {

                    "present":
                        True,

                    "valid":
                        False,

                    "feature_count":
                        len(
                            dns_features
                        ),

                    "expected_feature_count":
                        DNS_FEATURE_COUNT,

                    "reason":
                        (
                            "DNS schema validator returned "
                            "an invalid validation result."
                        ),
                }

            validation.setdefault(
                "present",
                True,
            )

            validation.setdefault(
                "feature_count",
                len(
                    dns_features
                ),
            )

            validation.setdefault(
                "expected_feature_count",
                DNS_FEATURE_COUNT,
            )

            return validation

        except Exception as exc:

            return {

                "present":
                    True,

                "valid":
                    False,

                "feature_count":
                    len(
                        dns_features
                    ),

                "expected_feature_count":
                    DNS_FEATURE_COUNT,

                "reason":
                    (
                        "DNS feature schema validation "
                        f"raised {type(exc).__name__}: {exc}"
                    ),
            }


    # ========================================================================
    # RUN SINGLE AGENT
    # ========================================================================

    async def _run_agent(
        self,
        agent_name: str,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute one agent safely.

        IMPORTANT FIX:

        The orchestrator no longer performs a generic pre-check such as:

            if feature block missing:
                return unavailable

        Instead, the COMPLETE unified vector is always passed to the agent.

        This prevents the orchestrator from blocking agents that can derive
        or process their own feature data internally.
        """

        started = time.perf_counter()

        configuration = self.agents.get(
            agent_name
        )

        if not configuration:

            return self._build_unavailable_result(
                agent_name=agent_name,
                feature_key=FEATURE_MAPPING.get(
                    agent_name,
                    "",
                ),
                reason=(
                    "Agent is not registered "
                    "in the orchestrator."
                ),
                started=started,
            )

        instance = configuration.get(
            "instance"
        )

        method_name = configuration.get(
            "method",
            "analyze",
        )

        feature_key = configuration.get(
            "feature_key"
        )

        try:

            if instance is None:

                raise RuntimeError(
                    "Agent instance is not configured."
                )

            method = getattr(
                instance,
                method_name,
                None,
            )

            if not callable(
                method
            ):

                raise AttributeError(
                    (
                        f"Agent '{agent_name}' does not expose "
                        f"a callable '{method_name}' method."
                    )
                )

            # ----------------------------------------------------------------
            # DNS VALIDATION INFORMATION
            #
            # We validate a supplied DNS block for diagnostics, but we do not
            # prevent the DNS agent from receiving the full vector when the
            # block is absent.
            # ----------------------------------------------------------------

            dns_validation = None

            if (
                agent_name
                == DNS_AGENT_NAME
            ):

                dns_validation = (
                    self._validate_dns_feature_block(
                        unified_vector
                    )
                )

                # ------------------------------------------------------------
                # If DNS block is present and invalid, preserve the diagnostic
                # but still let DNS agent receive the complete vector.
                #
                # This is important because the agent itself may support
                # extraction/default handling.
                # ------------------------------------------------------------

                if (
                    dns_validation.get(
                        "present",
                        False,
                    )
                    and
                    dns_validation.get(
                        "valid"
                    )
                    is False
                ):

                    logger.warning(
                        "DNS feature block validation warning: %s",
                        dns_validation.get(
                            "reason"
                        ),
                    )

            # ----------------------------------------------------------------
            # FEATURE AVAILABILITY DIAGNOSTIC
            # ----------------------------------------------------------------

            feature_block = unified_vector.get(
                feature_key
            )

            feature_block_present = (
                isinstance(
                    feature_block,
                    dict,
                )
                and bool(
                    feature_block
                )
            )

            # ----------------------------------------------------------------
            # EXECUTE AGENT (Performance Optimized & Timeout Protected)
            #
            # THE COMPLETE VECTOR IS PASSED.
            # ----------------------------------------------------------------

            timeout_sec = PerformanceConfig.AGENT_TIMEOUTS.get(agent_name, 10.0)

            async def _execute_with_concurrency():
                if inspect.iscoroutinefunction(method):
                    return await method(unified_vector)
                else:
                    return await asyncio.to_thread(method, unified_vector)

            try:
                result = await asyncio.wait_for(_execute_with_concurrency(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent_name} timed out after {timeout_sec} seconds")
                return self._build_unavailable_result(
                    agent_name=agent_name,
                    feature_key=feature_key,
                    reason=f"Agent timed out after {timeout_sec} seconds.",
                    started=started,
                )
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                return self._build_unavailable_result(
                    agent_name=agent_name,
                    feature_key=feature_key,
                    reason=f"Agent failed during execution.",
                    started=started,
                )

            # ----------------------------------------------------------------
            # Validate result type
            # ----------------------------------------------------------------

            if not isinstance(
                result,
                dict,
            ):

                return self._build_unavailable_result(
                    agent_name=agent_name,
                    feature_key=feature_key,
                    reason=(
                        "Agent returned a non-dictionary result."
                    ),
                    started=started,
                )

            # ----------------------------------------------------------------
            # Normalize result
            # ----------------------------------------------------------------

            normalized = (
                self._normalize_agent_result(
                    agent_name,
                    feature_key,
                    result,
                )
            )

            normalized[
                "feature_block_present"
            ] = feature_block_present

            normalized[
                "execution_time_ms"
            ] = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000.0,
                3,
            )

            if dns_validation is not None:

                normalized[
                    "dns_validation"
                ] = dns_validation

            return normalized

        except Exception as exc:

            logger.error(
                "Agent '%s' execution failed: %s",
                agent_name,
                exc,
                exc_info=True,
            )

            return self._build_unavailable_result(
                agent_name=agent_name,
                feature_key=feature_key,
                reason=(
                    f"Agent execution failed: {exc}"
                ),
                started=started,
                error=exc,
            )


    # ========================================================================
    # RESULT NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_agent_result(
        agent_name: str,
        feature_key: str,
        result: Any,
    ) -> Dict[str, Any]:
        """
        Normalize any agent response into the common orchestrator contract.

        Compatible with the Day 13 AgentResult-style fields and with the
        existing dictionary responses returned by all six agents.
        """

        if isinstance(
            result,
            dict,
        ):

            normalized = dict(
                result
            )

        else:

            normalized = {
                "result":
                    result,
            }

        # --------------------------------------------------------------------
        # Identity
        # --------------------------------------------------------------------

        normalized.setdefault(
            "agent",
            agent_name,
        )

        normalized.setdefault(
            "agent_name",
            agent_name,
        )

        normalized.setdefault(
            "feature_key",
            feature_key,
        )

        # --------------------------------------------------------------------
        # Status
        # --------------------------------------------------------------------

        status = str(
            normalized.get(
                "analysis_status",
                normalized.get(
                    "status",
                    "success",
                ),
            )
        ).lower()

        if status in {
            "failed",
            "failure",
        }:

            normalized[
                "analysis_status"
            ] = "error"

        else:

            normalized.setdefault(
                "analysis_status",
                status,
            )

        # --------------------------------------------------------------------
        # Signal availability
        # --------------------------------------------------------------------

        explicit_signal = (
            normalized.get(
                "signal_available"
            )
        )

        if explicit_signal is None:

            prediction = str(
                normalized.get(
                    "prediction",
                    normalized.get(
                        "verdict",
                        "unknown",
                    ),
                )
            ).lower()

            signal_available = (
                status
                not in {
                    "error",
                    "failed",
                    "failure",
                    "unavailable",
                    "no_signal",
                }
                and
                prediction
                not in {
                    "",
                    "unknown",
                    "none",
                    "null",
                }
            )

        else:

            signal_available = bool(
                explicit_signal
            )

        normalized[
            "signal_available"
        ] = signal_available

        # --------------------------------------------------------------------
        # Status consistency
        # --------------------------------------------------------------------

        if not signal_available:

            if normalized.get(
                "analysis_status"
            ) not in {
                "error",
                "unavailable",
                "no_signal",
            }:

                # Do not turn a missing signal into success.
                normalized[
                    "analysis_status"
                ] = "unavailable"

        else:

            normalized.setdefault(
                "analysis_status",
                "success",
            )

        normalized.setdefault(
            "status",
            normalized.get(
                "analysis_status",
                "success",
            ),
        )

        # --------------------------------------------------------------------
        # Safe defaults for common AgentResult fields
        # --------------------------------------------------------------------

        normalized.setdefault(
            "prediction",
            normalized.get(
                "verdict",
                "unknown",
            ),
        )

        # --------------------------------------------------------------------
        # Prediction vocabulary translation
        #
        # Each agent may use its own internal verdict labels.
        # Fusion's _is_usable_decision() only accepts:
        #   "legitimate", "phishing", "suspicious"
        # Translate known agent-specific labels to Fusion vocabulary here.
        # --------------------------------------------------------------------

        _PREDICTION_ALIASES: Dict[str, str] = {
            # Threat Intelligence Agent
            "malicious_threat": "phishing",
            "clean_reputation": "legitimate",

            # Generic aliases (belt-and-suspenders)
            "malicious": "phishing",
            "phish": "phishing",
            "phishing-url": "phishing",
            "phishing_site": "phishing",
            "safe": "legitimate",
            "benign": "legitimate",
            "clean": "legitimate",
            "medium": "suspicious",
            "warning": "suspicious",
        }

        raw_pred = str(
            normalized.get("prediction", "unknown")
        ).strip().lower()

        if raw_pred in _PREDICTION_ALIASES:
            normalized["prediction"] = (
                _PREDICTION_ALIASES[raw_pred]
            )


        normalized.setdefault(
            "probability",
            None,
        )

        # Confidence normalization.
        # 0 is treated as unavailable and recovered from sibling fields
        # when the agent actually produced a class probability.
        normalized["confidence"] = (
            MultiAgentOrchestrator._resolve_confidence(
                normalized
            )
        )

        if (
            signal_available
            and float(normalized.get("confidence") or 0.0) <= 0.0
            and str(normalized.get("prediction", "")).lower()
            in {"phishing", "legitimate", "suspicious", "malicious_threat", "clean_reputation"}
        ):

            # A classifier vote with no recoverable certainty is not a
            # usable success signal.
            normalized["signal_available"] = False
            normalized["analysis_status"] = "unavailable"
            normalized["status"] = "unavailable"
            normalized["prediction"] = "unknown"
            normalized["verdict"] = "unknown"
            if not normalized.get("reason"):
                normalized["reason"] = (
                    "Agent confidence was unavailable, so the "
                    "prediction was not published as a usable signal."
                )

        normalized.setdefault(
            "risk_score",
            None,
        )

        normalized.setdefault(
            "risk_level",
            "unknown",
        )

        normalized.setdefault(
            "prediction_source",
            "unavailable"
            if not signal_available
            else "unknown",
        )

        normalized.setdefault(
            "model_status",
            "unavailable"
            if not signal_available
            else "unknown",
        )

        normalized.setdefault(
            "risk_factors",
            [],
        )

        normalized.setdefault(
            "evidence",
            [],
        )

        normalized.setdefault(
            "explanation",
            {},
        )

        reason_candidate = (
            MultiAgentOrchestrator._resolve_reason(
                normalized
            )
        )

        if reason_candidate:
            normalized["reason"] = reason_candidate
        else:
            normalized.setdefault(
                "reason",
                None,
            )

        return normalized

    @staticmethod
    def _as_unit_interval(value: Any) -> Optional[float]:
        """
        Convert a numeric confidence/probability into [0, 1].
        """

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if numeric != numeric:
            return None

        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0

        if numeric < 0.0 or numeric > 1.0:
            return None

        return numeric

    @staticmethod
    def _resolve_confidence(
        payload: Dict[str, Any],
    ) -> float:
        """
        Recover class-certainty from explicit or derived agent fields.

        0 means confidence is unavailable, not "the model is sure".
        """

        for key in (
            "confidence",
            "confidence_score",
            "final_confidence",
        ):

            numeric = MultiAgentOrchestrator._as_unit_interval(
                payload.get(key)
            )

            if numeric is not None and numeric > 0.0:
                return round(numeric, 6)

        class_probabilities = payload.get(
            "class_probabilities"
        )

        prediction = str(
            payload.get("prediction")
            or payload.get("verdict")
            or ""
        ).strip().lower()

        if isinstance(class_probabilities, dict) and class_probabilities:

            if prediction in {
                "phishing",
                "malicious_threat",
                "malicious",
            }:
                for key in (
                    "malicious_threat",
                    "phishing",
                    "malicious",
                    "1",
                ):
                    numeric = MultiAgentOrchestrator._as_unit_interval(
                        class_probabilities.get(key)
                    )
                    if numeric is not None and numeric > 0.0:
                        return round(numeric, 6)

            if prediction in {
                "legitimate",
                "clean_reputation",
                "clean",
                "benign",
            }:
                for key in (
                    "clean_reputation",
                    "legitimate",
                    "clean",
                    "0",
                ):
                    numeric = MultiAgentOrchestrator._as_unit_interval(
                        class_probabilities.get(key)
                    )
                    if numeric is not None and numeric > 0.0:
                        return round(numeric, 6)

            values = []

            for value in class_probabilities.values():

                numeric = MultiAgentOrchestrator._as_unit_interval(
                    value
                )

                if numeric is not None:
                    values.append(numeric)

            if values:
                derived = max(values)
                if derived > 0.0:
                    return round(derived, 6)

        for key in (
            "threat_probability",
            "phishing_probability",
            "probability",
        ):

            numeric = MultiAgentOrchestrator._as_unit_interval(
                payload.get(key)
            )

            if numeric is None:
                continue

            if prediction in {
                "phishing",
                "malicious_threat",
                "malicious",
            }:
                derived = numeric
            elif prediction in {
                "legitimate",
                "clean_reputation",
                "clean",
                "benign",
            }:
                derived = 1.0 - numeric
            else:
                derived = max(numeric, 1.0 - numeric)

            if derived > 0.0:
                return round(derived, 6)

        return 0.0

    @staticmethod
    def _resolve_reason(
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        Always prefer a human-readable reason string when one exists.
        """

        candidates = [
            payload.get("reason"),
            payload.get("error"),
            payload.get("error_message"),
        ]

        explanation = payload.get("explanation")

        if isinstance(explanation, str):
            candidates.append(explanation)

        elif isinstance(explanation, dict):
            candidates.append(explanation.get("summary"))
            candidates.append(explanation.get("reason"))

        for candidate in candidates:

            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return None


    # ========================================================================
    # ASYNC AGENT DISPATCH
    # ========================================================================

    async def _run_all_agents_async(
        self,
        unified_vector: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute all active agents concurrently.
        """

        tasks = [

            self._run_agent(
                agent_name,
                unified_vector,
            )

            for agent_name
            in ACTIVE_AGENT_NAMES
        ]

        if not tasks:

            return []

        results = await asyncio.gather(
            *tasks,
            return_exceptions=False,
        )

        return list(
            results
        )


    # ========================================================================
    # SYNC ASYNC BRIDGE
    # ========================================================================

    def _execute_all_agents(
        self,
        unified_vector: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute the asynchronous agent pipeline from synchronous code.
        """

        try:

            asyncio.get_running_loop()

        except RuntimeError:

            return asyncio.run(
                self._run_all_agents_async(
                    unified_vector
                )
            )

        # --------------------------------------------------------------------
        # If an event loop is already running, execute the coroutine in a
        # dedicated worker thread.
        # --------------------------------------------------------------------

        import threading

        result_holder = {
            "results":
                None,

            "error":
                None,
        }

        def runner() -> None:

            try:

                result_holder[
                    "results"
                ] = asyncio.run(
                    self._run_all_agents_async(
                        unified_vector
                    )
                )

            except Exception as exc:

                result_holder[
                    "error"
                ] = exc

        thread = threading.Thread(
            target=runner,
            daemon=True,
        )

        thread.start()
        thread.join()

        if result_holder[
            "error"
        ] is not None:

            raise result_holder[
                "error"
            ]

        return (
            result_holder[
                "results"
            ]
            or []
        )


    # ========================================================================
    # BUILD FUSION SIGNALS
    # ========================================================================

    @staticmethod
    def _build_fusion_signals(
        agent_results: List[
            Dict[str, Any]
        ],
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:
        """
        Extract only usable agent results for Fusion.

        Unavailable/error/unknown signals are never promoted to legitimate
        evidence.
        """

        fusion_signals = {}

        for result in agent_results:

            if not isinstance(
                result,
                dict,
            ):

                continue

            agent_name = (
                result.get(
                    "agent_name"
                )
                or result.get(
                    "agent"
                )
            )

            if (
                agent_name
                not in ACTIVE_AGENT_NAMES
            ):

                continue

            status = str(
                result.get(
                    "analysis_status",
                    result.get(
                        "status",
                        "",
                    ),
                )
            ).lower()

            signal_available = bool(
                result.get(
                    "signal_available",
                    False,
                )
            )

            prediction = str(
                result.get(
                    "prediction",
                    result.get(
                        "verdict",
                        "unknown",
                    ),
                )
            ).lower()

            if status in {
                "error",
                "failed",
                "failure",
                "unavailable",
                "no_signal",
            }:

                continue

            if not signal_available:

                continue

            if prediction in {
                "",
                "unknown",
                "none",
                "null",
            }:

                continue

            fusion_signals[
                agent_name
            ] = result

        return fusion_signals


    # ========================================================================
    # FUSION FALLBACK
    # ========================================================================

    def _build_fusion_error_result(
        self,
        error: Exception,
        fusion_signals: Dict[
            str,
            Dict[str, Any],
        ],
    ) -> Dict[str, Any]:
        """
        Build a stable Fusion contract when Fusion itself fails.
        """

        usable_agents = sorted(
            fusion_signals.keys()
        )

        excluded_agents = sorted(
            [
                agent

                for agent
                in ACTIVE_AGENT_NAMES

                if agent
                not in fusion_signals
            ]
        )

        return {

            "engine_name":
                getattr(
                    self.fusion_engine,
                    "engine_name",
                    "Decision Fusion Engine",
                ),

            "engine_version":
                getattr(
                    self.fusion_engine,
                    "engine_version",
                    "17.0.0",
                ),

            "status":
                "error",

            "analysis_status":
                "error",

            "signal_available":
                bool(
                    fusion_signals
                ),

            "final_verdict":
                "unknown",

            "verdict":
                "unknown",

            "risk_level":
                "unknown",

            "final_risk_level":
                "unknown",

            "final_risk_score":
                None,

            "risk_score":
                None,

            "confidence":
                0.0,

            "phishing_probability":
                None,

            "final_phishing_probability":
                None,

            "fused_probability":
                None,

            "legitimate_probability":
                None,

            "usable_agent_count":
                len(
                    usable_agents
                ),

            "usable_agents":
                usable_agents,

            "excluded_agents":
                excluded_agents,

            "consensus_available":
                False,

            "consensus_satisfied":
                False,

            "limited_evidence":
                len(
                    usable_agents
                )
                < 2,

            "consensus":
                {},

            "agent_contributions":
                [],

            "html_agent_contribution":
                None,

            "threat_intelligence":
                None,

            "dns_critical_evidence":
                None,

            "critical_evidence_override":
                False,

            "critical_evidence":
                {
                    "critical":
                        False,

                    "critical_sources":
                        [],

                    "reasons":
                        [],
                },

            "conflict_detected":
                False,

            "conflict_level":
                "none",

            "conflicting_agents":
                [],

            "conflict_reasons":
                [],

            "evidence_state":
                "unavailable",

            "decision_basis":
                {
                    "type":
                        "fusion_error",

                    "critical_override":
                        False,

                    "reasons":
                        [
                            str(
                                error
                            )
                        ],

                    "usable_agent_count":
                        len(
                            usable_agents
                        ),
                },

            "weighting":
                {},

            "reason":
                "Decision Fusion execution failed.",

            "error":
                str(
                    error
                ),

            "error_type":
                type(
                    error
                ).__name__,
        }


    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        unified_vector: Any,
    ) -> Dict[str, Any]:
        """
        Execute the complete multi-agent cybersecurity analysis pipeline.

        Accepts:
            unified_vector (dict) - Full or partial unified feature vector.
            unified_vector (str)  - Raw URL string. The orchestrator wraps it
                                    into {"url": ..., "domain": ..., "metadata": ...}
                                    automatically so feature-block agents will
                                    run in URL-only mode (others mark unavailable).
        """

        started = time.perf_counter()

        # ====================================================================
        # INPUT COERCION — Accept raw URL string
        # ====================================================================

        if isinstance(unified_vector, str):

            raw_url = unified_vector.strip()

            try:
                from urllib.parse import urlparse as _urlparse
                parsed = _urlparse(raw_url)
                domain = parsed.netloc or parsed.path
            except Exception:
                domain = raw_url

            unified_vector = {
                "url": raw_url,
                "domain": domain,
                "metadata": {
                    "target_url": raw_url,
                },
            }

        # ====================================================================
        # INPUT VALIDATION
        # ====================================================================

        if not isinstance(
            unified_vector,
            dict,
        ):

            return self._build_input_error_result(
                "Unified vector must be a dictionary or a URL string."
            )

        # ====================================================================
        # COPY INPUT
        #
        # Do not mutate caller-owned data.
        # ====================================================================

        vector = dict(
            unified_vector
        )

        # ====================================================================
        # EXECUTE ALL AGENTS
        # ====================================================================

        try:

            agent_results = (
                self._execute_all_agents(
                    vector
                )
            )

        except Exception as exc:

            logger.error(
                "Global agent execution failure: %s",
                exc,
                exc_info=True,
            )

            agent_results = []

            for agent_name in ACTIVE_AGENT_NAMES:

                agent_results.append(
                    self._build_unavailable_result(
                        agent_name=agent_name,
                        feature_key=FEATURE_MAPPING.get(
                            agent_name,
                            "",
                        ),
                        reason=(
                            "Orchestrator agent execution "
                            f"failed: {exc}"
                        ),
                        started=started,
                        error=exc,
                    )
                )

        # ====================================================================
        # GUARANTEE SIX RESULT SLOTS
        # ====================================================================

        existing_agents = set()

        for result in agent_results:

            if isinstance(
                result,
                dict,
            ):

                name = (
                    result.get(
                        "agent_name"
                    )
                    or result.get(
                        "agent"
                    )
                )

                if name:

                    existing_agents.add(
                        name
                    )

        for agent_name in ACTIVE_AGENT_NAMES:

            if agent_name not in existing_agents:

                agent_results.append(
                    self._build_unavailable_result(
                        agent_name=agent_name,
                        feature_key=FEATURE_MAPPING.get(
                            agent_name,
                            "",
                        ),
                        reason=(
                            "Agent result was not produced."
                        ),
                        started=started,
                    )
                )

        # ====================================================================
        # NORMALIZE ORDER
        # ====================================================================

        ordered_results = []

        for agent_name in ACTIVE_AGENT_NAMES:

            matched = None

            for result in agent_results:

                if not isinstance(
                    result,
                    dict,
                ):

                    continue

                current_name = (
                    result.get(
                        "agent_name"
                    )
                    or result.get(
                        "agent"
                    )
                )

                if current_name == agent_name:

                    matched = result
                    break

            if matched is not None:

                ordered_results.append(
                    matched
                )

        agent_results = ordered_results

        # ====================================================================
        # AGENT STATUS CLASSIFICATION
        # ====================================================================

        successful_agents = []
        failed_agents = []
        unavailable_agents = []

        for result in agent_results:

            if not isinstance(
                result,
                dict,
            ):

                continue

            agent_name = (
                result.get(
                    "agent_name"
                )
                or result.get(
                    "agent"
                )
            )

            status = str(
                result.get(
                    "analysis_status",
                    result.get(
                        "status",
                        "",
                    ),
                )
            ).lower()

            signal_available = bool(
                result.get(
                    "signal_available",
                    False,
                )
            )

            if (
                status == "success"
                and signal_available
            ):

                successful_agents.append(
                    agent_name
                )

            elif status in {
                "error",
                "failed",
                "failure",
            }:

                failed_agents.append(
                    agent_name
                )

            else:

                unavailable_agents.append(
                    agent_name
                )

        # ====================================================================
        # DNS VALIDATION
        # ====================================================================

        dns_validation = (
            self._validate_dns_feature_block(
                vector
            )
        )

        # ====================================================================
        # THREAT INTELLIGENCE SIGNAL
        # ====================================================================

        threat_intelligence_signal = None

        for result in agent_results:

            if not isinstance(
                result,
                dict,
            ):

                continue

            agent_name = (
                result.get(
                    "agent_name"
                )
                or result.get(
                    "agent"
                )
            )

            if (
                agent_name
                != THREAT_INTEL_AGENT_NAME
            ):

                continue

            threat_intelligence_signal = (
                result.get(
                    "threat_intelligence"
                )
                or result.get(
                    "threat_intel"
                )
            )

            if threat_intelligence_signal is None:

                threat_intelligence_signal = result

            break

        # ====================================================================
        # FUSION SIGNALS
        # ====================================================================

        fusion_signals = (
            self._build_fusion_signals(
                agent_results
            )
        )

        # ====================================================================
        # EXECUTE FUSION
        # ====================================================================

        try:

            fusion_result = (
                self.fusion_engine.fuse(
                    agent_results
                )
            )

            if not isinstance(
                fusion_result,
                dict,
            ):

                raise TypeError(
                    "Fusion engine returned a non-dictionary result."
                )

        except Exception as exc:

            logger.error(
                "Decision Fusion execution failed: %s",
                exc,
                exc_info=True,
            )

            fusion_result = (
                self._build_fusion_error_result(
                    exc,
                    fusion_signals,
                )
            )

        # ====================================================================
        # EXECUTION TIME
        # ====================================================================

        execution_time_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000.0,
            3,
        )

        # ====================================================================
        # FUSION USABLE AGENTS
        # ====================================================================

        fusion_usable_agents = fusion_result.get(
            "usable_agents",
            sorted(
                fusion_signals.keys()
            ),
        )

        fusion_usable_count = fusion_result.get(
            "usable_agent_count",
            len(
                fusion_usable_agents
            ),
        )

        # ====================================================================
        # EXCLUDED AGENTS
        # ====================================================================

        excluded_agents = fusion_result.get(
            "excluded_agents",
            sorted(
                [
                    agent

                    for agent
                    in ACTIVE_AGENT_NAMES

                    if agent
                    not in set(
                        fusion_usable_agents
                    )
                ]
            ),
        )

        # ====================================================================
        # TOP-LEVEL OUTPUT
        # ====================================================================

        final_verdict = fusion_result.get(
            "final_verdict",
            fusion_result.get(
                "verdict",
                "unknown",
            ),
        )

        final_risk_score = fusion_result.get(
            "final_risk_score"
        )

        risk_level = fusion_result.get(
            "risk_level",
            fusion_result.get(
                "final_risk_level",
                "unknown",
            ),
        )

        confidence = fusion_result.get(
            "confidence",
            0.0,
        )

        signal_available = bool(
            fusion_result.get(
                "signal_available",
                bool(
                    fusion_signals
                ),
            )
        )

        # ====================================================================
        # ANALYSIS STATUS
        # ====================================================================

        fusion_status = str(
            fusion_result.get(
                "analysis_status",
                fusion_result.get(
                    "status",
                    "success"
                    if signal_available
                    else "no_signal",
                ),
            )
        )

        # ====================================================================
        # RESULT
        # ====================================================================

        return {

            "orchestrator":
                self.orchestrator_name,

            "orchestrator_version":
                self.orchestrator_version,

            "analysis_status":
                fusion_status,

            "signal_available":
                signal_available,

            "execution_time_ms":
                execution_time_ms,

            # ----------------------------------------------------------------
            # Agent information
            # ----------------------------------------------------------------

            "active_agents":
                list(
                    ACTIVE_AGENT_NAMES
                ),

            "agent_count":
                len(
                    ACTIVE_AGENT_NAMES
                ),

            "successful_agents":
                successful_agents,

            "failed_agents":
                failed_agents,

            "unavailable_agents":
                unavailable_agents,

            "usable_agents":
                fusion_usable_agents,

            "usable_agent_count":
                fusion_usable_count,

            "excluded_agents":
                excluded_agents,

            # ----------------------------------------------------------------
            # Agent-specific validation
            # ----------------------------------------------------------------

            "html_validation":
                self._validate_named_agent_signal(
                    agent_results,
                    HTML_AGENT_NAME,
                ),

            "ssl_validation":
                self._validate_named_agent_signal(
                    agent_results,
                    SSL_AGENT_NAME,
                ),

            "dns_validation":
                dns_validation,

            # ----------------------------------------------------------------
            # Raw agent outputs
            # ----------------------------------------------------------------

            "agent_results":
                agent_results,

            "agents":
                agent_results,

            # ----------------------------------------------------------------
            # Fusion signals
            # ----------------------------------------------------------------

            "fusion_signals":
                fusion_signals,

            "threat_intelligence":
                threat_intelligence_signal,

            "fusion":
                fusion_result,

            "fusion_result":
                fusion_result,

            # ----------------------------------------------------------------
            # Final decision
            # ----------------------------------------------------------------

            "final_verdict":
                final_verdict,

            "verdict":
                final_verdict,

            "risk_score":
                final_risk_score,

            "final_risk_score":
                final_risk_score,

            "risk_level":
                risk_level,

            "confidence":
                confidence,

            "phishing_probability":
                fusion_result.get(
                    "phishing_probability"
                ),

            "legitimate_probability":
                fusion_result.get(
                    "legitimate_probability"
                ),

            # ----------------------------------------------------------------
            # Consensus
            # ----------------------------------------------------------------

            "consensus":
                fusion_result.get(
                    "consensus",
                    {},
                ),

            "consensus_available":
                fusion_result.get(
                    "consensus_available",
                    False,
                ),

            # ----------------------------------------------------------------
            # Decision basis
            # ----------------------------------------------------------------

            "decision_basis":
                fusion_result.get(
                    "decision_basis",
                    {},
                ),

            # ----------------------------------------------------------------
            # Conflict & evidence state (Required top-level contract)
            # ----------------------------------------------------------------

            "conflict_detected":
                fusion_result.get(
                    "conflict_detected",
                    False,
                ),

            "evidence_state":
                fusion_result.get(
                    "evidence_state",
                    "unavailable",
                ),

            "critical_evidence_override":
                fusion_result.get(
                    "critical_evidence_override",
                    False,
                ),

            "weighting":
                fusion_result.get(
                    "weighting",
                    {},
                ),

            "reason":
                fusion_result.get(
                    "reason"
                ),

            # ----------------------------------------------------------------
            # Integration consistency
            # ----------------------------------------------------------------

            "agent_status_consistency":
                {
                    "successful_agent_count":
                        len(
                            successful_agents
                        ),

                    "failed_agent_count":
                        len(
                            failed_agents
                        ),

                    "unavailable_agent_count":
                        len(
                            unavailable_agents
                        ),

                    "fusion_usable_agent_count":
                        fusion_usable_count,

                    "consistent":
                        set(
                            successful_agents
                        ).issuperset(
                            set(
                                fusion_usable_agents
                            )
                        ),
                },
        }


    # ========================================================================
    # NAMED AGENT VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_named_agent_signal(
        agent_results: List[
            Dict[str, Any]
        ],
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        Return lightweight validation information for a named agent.
        """

        for result in agent_results:

            if not isinstance(
                result,
                dict,
            ):

                continue

            current_name = (
                result.get(
                    "agent_name"
                )
                or result.get(
                    "agent"
                )
            )

            if current_name != agent_name:

                continue

            return {

                "present":
                    True,

                "analysis_status":
                    result.get(
                        "analysis_status"
                    ),

                "signal_available":
                    bool(
                        result.get(
                            "signal_available",
                            False,
                        )
                    ),

                "feature_key":
                    result.get(
                        "feature_key"
                    ),

                "feature_block_present":
                    result.get(
                        "feature_block_present"
                    ),

                "reason":
                    result.get(
                        "reason"
                    ),

                "error":
                    result.get(
                        "error"
                    ),

                "dns_validation":
                    result.get(
                        "dns_validation"
                    ),
            }

        return {

            "present":
                False,

            "analysis_status":
                "unavailable",

            "signal_available":
                False,

            "feature_key":
                FEATURE_MAPPING.get(
                    agent_name
                ),

            "feature_block_present":
                False,

            "reason":
                "Agent result was not produced.",

            "error":
                None,
        }


    # ========================================================================
    # PARALLEL ANALYSIS
    # ========================================================================

    def analyze_parallel(
        self,
        unified_vector: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compatibility entry point used by app.py.

        Agent execution is already parallelized internally by analyze().
        """

        return self.analyze(
            unified_vector
        )


    # ========================================================================
    # MODEL RELOAD
    # ========================================================================

    def reload_model(
        self,
    ) -> Dict[str, Any]:
        """
        Reload models for all active agents when supported.
        """

        results = {}

        for agent_name in ACTIVE_AGENT_NAMES:

            registry = self.agents.get(
                agent_name
            )

            if not registry:

                results[
                    agent_name
                ] = {
                    "status":
                        "not_registered",
                }

                continue

            instance = registry.get(
                "instance"
            )

            reload_method = getattr(
                instance,
                "reload_model",
                None,
            )

            if not callable(
                reload_method
            ):

                results[
                    agent_name
                ] = {

                    "status":
                        "not_supported",
                }

                continue

            try:

                result = reload_method()

                results[
                    agent_name
                ] = {

                    "status":
                        "success",

                    "result":
                        result,
                }

            except Exception as exc:

                results[
                    agent_name
                ] = {

                    "status":
                        "error",

                    "error":
                        str(exc),

                    "error_type":
                        type(exc).__name__,
                }

        return results


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def analyze_unified_vector(
    unified_vector: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience wrapper for one-shot unified-vector analysis.
    """

    orchestrator = (
        MultiAgentOrchestrator()
    )

    return orchestrator.analyze(
        unified_vector
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [

    "MultiAgentOrchestrator",

    "analyze_unified_vector",

    "ACTIVE_AGENT_NAMES",

    "FEATURE_MAPPING",

    "ORCHESTRATOR_NAME",

    "ORCHESTRATOR_VERSION",

    "URL_AGENT_NAME",

    "HTML_AGENT_NAME",

    "SSL_AGENT_NAME",

    "DNS_AGENT_NAME",

    "VISUAL_AGENT_NAME",

    "THREAT_INTEL_AGENT_NAME",
]