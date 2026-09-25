import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Database, 
  Server, 
  Cpu, 
  ShieldCheck, 
  Code, 
  CheckCircle2, 
  RefreshCw,
  Sliders,
  FileCode,
  Terminal
} from 'lucide-react';
import { checkHealth } from '../services/api';

export default function SettingsPage() {
  const [health, setHealth] = useState(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    runPing();
  }, []);

  const runPing = async () => {
    setTesting(true);
    const res = await checkHealth();
    setHealth(res);
    setTesting(false);
  };

  const agentRegistry = [
    {
      id: 'url',
      name: 'URL AI Agent',
      model: 'XGBoost Lexical Classifier',
      features: '11 Lexical & Syntactic Features',
      curve: 'url_lexical_probability_mapping',
      rationale: 'Direct lexical attack probability mapped linearly to operational severity (0-100).'
    },
    {
      id: 'html',
      name: 'HTML AI Agent',
      model: 'DOM Tree Structure & Forms Engine',
      features: '50 Structural DOM Features',
      curve: 'html_dom_structure_mapping',
      rationale: 'DOM structure and form analysis mapped to operational severity with credential weighting.'
    },
    {
      id: 'ssl',
      name: 'SSL AI Agent',
      model: 'X.509 Cryptographic Validator',
      features: '17 Cryptographic Validation Points',
      curve: 'ssl_cryptographic_mapping',
      rationale: 'SSL/TLS certificate authority, validity period and cipher suites operational severity.'
    },
    {
      id: 'dns',
      name: 'DNS AI Agent',
      model: 'XGBoost Network Telemetry Classifier',
      features: '35 Canonical DNS Features',
      curve: 'dns_legitimate_baseline (infrastructure floor)',
      rationale: 'Harmonizes high legitimate infrastructure scores to ~33.75 while maintaining 1:1 parity on malicious indicators (DNS 70 == URL 70).'
    },
    {
      id: 'visual',
      name: 'Visual AI Agent',
      model: 'Perceptual Hash & Brand Similarity',
      features: '12 Brand Signature Vectors',
      curve: 'visual_high_confidence_direct_mapping',
      rationale: 'High-confidence brand/layout phishing indicators mapped directly to severe operational risk.'
    },
    {
      id: 'threat',
      name: 'Threat Intelligence Agent',
      model: 'Global IOC Correlator',
      features: 'Multi-Feed IOC Reputation Telemetry',
      curve: 'threat_intel_standard_mapping',
      rationale: 'Step-threshold mapping ensuring confirmed threat feed entries instantly trigger critical security posture.'
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Backend & Database Diagnostics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
        {/* API Diagnostics */}
        <div style={{
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Server size={18} color="var(--cyber-blue)" />
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
                Backend API Infrastructure
              </h3>
            </div>
            <button
              onClick={runPing}
              disabled={testing}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '4px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                color: '#f8fafc',
                fontSize: '11px',
                cursor: 'pointer'
              }}
            >
              <RefreshCw size={12} style={{ animation: testing ? 'spin 1s linear infinite' : 'none' }} />
              <span>Ping :5050</span>
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Endpoint Address</span>
              <span style={{ fontFamily: 'monospace', color: 'var(--cyber-blue)' }}>http://127.0.0.1:5050</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Liveness Status</span>
              <span style={{ color: health?.status === 'healthy' ? '#34d399' : '#fb7185', fontWeight: 700 }}>
                {health?.status?.toUpperCase() || 'OFFLINE'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Orchestrator Engine</span>
              <span style={{ color: '#f8fafc' }}>{health?.orchestrator || 'Multi-Agent Cybersecurity Orchestrator'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Fusion Version</span>
              <span style={{ fontFamily: 'monospace', color: '#f8fafc' }}>{health?.fusion_version || '17.0.0'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Active Agent Capacity</span>
              <span style={{ fontWeight: 700, color: 'var(--cyber-blue)' }}>{health?.active_agents ?? 6} / 6 Agents</span>
            </div>
          </div>
        </div>

        {/* Database Diagnostics */}
        <div style={{
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={18} color="var(--cyber-blue)" />
            <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f8fafc' }}>
              SQLite Persistence Engine
            </h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Target Database Path</span>
              <span style={{ fontFamily: 'monospace', color: 'var(--cyber-blue)' }}>data/phishdec.db</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Schema Tables</span>
              <span style={{ color: '#f8fafc', fontWeight: 600 }}>analyses, agent_results, risk_factors, evidence</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Referential Integrity</span>
              <span style={{ color: '#34d399', fontWeight: 600 }}>PRAGMA foreign_keys = ON</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Deletion Policy</span>
              <span style={{ color: '#34d399', fontWeight: 600 }}>CASCADE (No Orphan Records)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'rgba(6,9,15,0.4)', borderRadius: '4px' }}>
              <span style={{ color: 'var(--text-faint)' }}>Null Safety Guarantee</span>
              <span style={{ color: '#34d399', fontWeight: 600 }}>Strict SQL NULL (No Zero-Coercion)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 6 AI Agents Specification Registry */}
      <div style={{
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '8px',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#f8fafc', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={18} color="var(--cyber-blue)" />
            <span>6-Agent Swarm Registry & Calibration Specifications</span>
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Each agent operates with its own distinct feature extractor, prediction engine, and Operational Security Risk Harmonization (OSRH-v2) curve.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px' }}>
          {agentRegistry.map((agent) => (
            <div
              key={agent.id}
              style={{
                backgroundColor: 'rgba(6, 9, 15, 0.5)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '6px',
                padding: '16px 18px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                fontSize: '12px'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '13px' }}>{agent.name}</span>
                <span style={{ fontSize: '10px', color: '#34d399', fontWeight: 700, padding: '2px 6px', backgroundColor: 'rgba(52, 211, 153, 0.1)', borderRadius: '3px' }}>
                  ACTIVE
                </span>
              </div>
              <div style={{ color: 'var(--cyber-blue)', fontSize: '11px', fontFamily: 'monospace' }}>
                {agent.model}
              </div>
              <div style={{ color: 'var(--text-faint)', fontSize: '11px' }}>
                Vector: {agent.features}
              </div>
              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '8px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                {agent.rationale}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-faint)', fontFamily: 'monospace' }}>
                Curve: {agent.curve}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
